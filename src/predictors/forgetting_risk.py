"""Leakage-safe forgetting-risk feature building and heuristic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PREDICTOR_REPORT_SCHEMA_VERSION = 1
PRIMARY_TARGET = "forgot_any_future"
FEATURE_COLUMNS = (
    "anchor_loss",
    "anchor_uncertainty",
    "anchor_confidence",
    "anchor_target_probability",
    "history_eval_count",
    "has_previous_eval",
    "previous_correct",
    "previous_loss",
    "previous_uncertainty",
    "previous_target_probability",
    "loss_delta_from_previous",
    "uncertainty_delta_from_previous",
    "target_probability_delta_from_previous",
    "loss_increase_from_previous",
    "target_probability_drop_from_previous",
    "max_loss_so_far",
    "min_target_probability_so_far",
    "max_uncertainty_so_far",
    "correct_history_rate",
    "tasks_since_source",
    "anchor_task_progress",
)
HEURISTIC_SCORE_NAMES = (
    "anchor_loss",
    "anchor_uncertainty",
    "low_target_probability",
    "loss_increase_from_previous",
    "target_probability_drop_from_previous",
    "combined_signal",
)


@dataclass(frozen=True)
class TemporalSplit:
    """Temporal anchor-task split used for predictor evaluation."""

    train_anchor_task_max: int
    test_anchor_task_min: int


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        newline="\n",
    ) as tmp_file:
        json.dump(payload, tmp_file, indent=2, sort_keys=True, default=_json_default)
        tmp_file.write("\n")
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(row: dict[str, Any], key: str) -> int:
    if key not in row:
        raise KeyError(f"row is missing required key {key!r}")
    return int(row[key])


def _as_float(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise KeyError(f"row is missing required key {key!r}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite, got {row[key]!r}")
    return value


def _as_bool(row: dict[str, Any], key: str) -> bool:
    if key not in row:
        raise KeyError(f"row is missing required key {key!r}")
    return bool(row[key])


def _signal_history(signal_payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    rows = signal_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("signal payload must contain rows")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("observation_type") != "seen_task_eval":
            continue
        grouped.setdefault(_as_int(row, "sample_id"), []).append(row)

    for sample_rows in grouped.values():
        sample_rows.sort(
            key=lambda row: (
                _as_int(row, "trained_task_id"),
                _as_int(row, "global_step"),
            )
        )
    return grouped


def _find_anchor_history(
    *,
    label_row: dict[str, Any],
    sample_history: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor_task = _as_int(label_row, "anchor_trained_task_id")
    anchor_step = _as_int(label_row, "anchor_global_step")
    anchor_eval_task = _as_int(label_row, "anchor_evaluated_task_id")

    history = [
        row
        for row in sample_history
        if (
            _as_int(row, "trained_task_id") < anchor_task
            or (
                _as_int(row, "trained_task_id") == anchor_task
                and _as_int(row, "global_step") <= anchor_step
            )
        )
    ]
    anchor_matches = [
        row
        for row in history
        if _as_int(row, "trained_task_id") == anchor_task
        and _as_int(row, "global_step") == anchor_step
        and _as_int(row, "evaluated_task_id") == anchor_eval_task
    ]
    if len(anchor_matches) != 1:
        raise ValueError(
            "could not uniquely match label anchor to signal row for "
            f"sample_id={label_row.get('sample_id')!r}, "
            f"anchor_task={anchor_task}, anchor_step={anchor_step}"
        )
    return anchor_matches[0], history


def _feature_row(
    *,
    label_row: dict[str, Any],
    signal_history: dict[int, list[dict[str, Any]]],
    max_anchor_task: int,
) -> dict[str, Any]:
    sample_id = _as_int(label_row, "sample_id")
    if sample_id not in signal_history:
        raise ValueError(f"missing signal history for sample_id={sample_id}")

    anchor, history = _find_anchor_history(
        label_row=label_row,
        sample_history=signal_history[sample_id],
    )
    if not history:
        raise ValueError(f"empty anchor history for sample_id={sample_id}")

    previous = history[-2] if len(history) >= 2 else None
    anchor_loss = _as_float(anchor, "loss")
    anchor_uncertainty = _as_float(anchor, "uncertainty")
    anchor_confidence = _as_float(anchor, "confidence")
    anchor_target_probability = _as_float(anchor, "target_probability")
    previous_loss = _as_float(previous, "loss") if previous is not None else anchor_loss
    previous_uncertainty = (
        _as_float(previous, "uncertainty") if previous is not None else anchor_uncertainty
    )
    previous_target_probability = (
        _as_float(previous, "target_probability")
        if previous is not None
        else anchor_target_probability
    )
    history_losses = [_as_float(row, "loss") for row in history]
    history_target_probabilities = [
        _as_float(row, "target_probability") for row in history
    ]
    history_uncertainties = [_as_float(row, "uncertainty") for row in history]
    correct_history = [_as_bool(row, "correct") for row in history]
    anchor_task = _as_int(label_row, "anchor_trained_task_id")
    source_task = _as_int(label_row, "source_task_id")
    loss_delta = anchor_loss - previous_loss
    target_probability_delta = anchor_target_probability - previous_target_probability

    row = {
        "sample_id": sample_id,
        "split": str(label_row["split"]),
        "source_task_id": source_task,
        "original_class_id": _as_int(label_row, "original_class_id"),
        "anchor_trained_task_id": anchor_task,
        "anchor_global_step": _as_int(label_row, "anchor_global_step"),
        "eligible_for_binary_forgetting": _as_bool(
            label_row,
            "eligible_for_binary_forgetting",
        ),
        "forgot_any_future": _as_bool(label_row, "forgot_any_future"),
        "forgot_next_eval": _as_bool(label_row, "forgot_next_eval"),
        "forgot_final_eval": _as_bool(label_row, "forgot_final_eval"),
        "anchor_loss": anchor_loss,
        "anchor_uncertainty": anchor_uncertainty,
        "anchor_confidence": anchor_confidence,
        "anchor_target_probability": anchor_target_probability,
        "history_eval_count": len(history),
        "has_previous_eval": int(previous is not None),
        "previous_correct": int(_as_bool(previous, "correct")) if previous is not None else int(_as_bool(anchor, "correct")),
        "previous_loss": previous_loss,
        "previous_uncertainty": previous_uncertainty,
        "previous_target_probability": previous_target_probability,
        "loss_delta_from_previous": loss_delta,
        "uncertainty_delta_from_previous": anchor_uncertainty - previous_uncertainty,
        "target_probability_delta_from_previous": target_probability_delta,
        "loss_increase_from_previous": max(0.0, loss_delta),
        "target_probability_drop_from_previous": max(0.0, -target_probability_delta),
        "max_loss_so_far": max(history_losses),
        "min_target_probability_so_far": min(history_target_probabilities),
        "max_uncertainty_so_far": max(history_uncertainties),
        "correct_history_rate": sum(correct_history) / len(correct_history),
        "tasks_since_source": anchor_task - source_task,
        "anchor_task_progress": (
            anchor_task / max_anchor_task if max_anchor_task > 0 else 0.0
        ),
        "feature_uses_rows_up_to_trained_task_id": anchor_task,
        "feature_uses_rows_up_to_global_step": _as_int(label_row, "anchor_global_step"),
        "leakage_safe": True,
    }
    for target_name in (
        "next_loss_delta",
        "final_loss_delta",
        "max_future_loss_increase",
        "next_target_probability_drop",
        "final_target_probability_drop",
        "max_future_target_probability_drop",
        "next_confidence_drop",
        "final_confidence_drop",
        "max_future_confidence_drop",
        "future_min_target_probability",
        "future_max_loss",
    ):
        if target_name in label_row and label_row[target_name] is not None:
            row[target_name] = _as_float(label_row, target_name)
    return row


def build_feature_rows(
    *,
    signal_payload: dict[str, Any],
    label_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build predictor rows using only at-anchor or pre-anchor signal history."""

    label_rows = label_payload.get("rows")
    if not isinstance(label_rows, list):
        raise ValueError("label payload must contain rows")
    if not label_rows:
        raise ValueError("label payload has no rows")

    signal_history = _signal_history(signal_payload)
    max_anchor_task = max(_as_int(row, "anchor_trained_task_id") for row in label_rows)
    rows = [
        _feature_row(
            label_row=row,
            signal_history=signal_history,
            max_anchor_task=max_anchor_task,
        )
        for row in label_rows
    ]
    if not all(row["leakage_safe"] for row in rows):
        raise ValueError("feature builder produced a non-leakage-safe row")
    return rows


def _default_split(feature_rows: list[dict[str, Any]]) -> TemporalSplit:
    anchor_tasks = sorted({int(row["anchor_trained_task_id"]) for row in feature_rows})
    if len(anchor_tasks) < 2:
        raise ValueError("temporal evaluation requires at least two anchor tasks")
    split_index = max(1, int(len(anchor_tasks) * 0.6)) - 1
    split_index = min(split_index, len(anchor_tasks) - 2)
    train_max = anchor_tasks[split_index]
    return TemporalSplit(
        train_anchor_task_max=train_max,
        test_anchor_task_min=anchor_tasks[split_index + 1],
    )


def _eligible_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in feature_rows if row["eligible_for_binary_forgetting"]]


def _score_array(rows: list[dict[str, Any]], score_name: str) -> np.ndarray:
    if score_name == "low_target_probability":
        return np.array([1.0 - row["anchor_target_probability"] for row in rows], dtype=float)
    if score_name == "combined_signal":
        raise ValueError("combined_signal must be scored with fitted scaler")
    return np.array([row[score_name] for row in rows], dtype=float)


def _fit_minmax(train_rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    fitted: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = _score_array(train_rows, key)
        fitted[key] = (float(np.min(values)), float(np.max(values)))
    return fitted


def _minmax_transform(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if maximum <= minimum:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _combined_signal_scores(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> np.ndarray:
    keys = (
        "anchor_loss",
        "anchor_uncertainty",
        "low_target_probability",
        "loss_increase_from_previous",
        "target_probability_drop_from_previous",
    )
    fitted = _fit_minmax(train_rows, keys)
    transformed = []
    for key in keys:
        raw = _score_array(test_rows, key)
        transformed.append(_minmax_transform(raw, *fitted[key]))
    return np.mean(np.vstack(transformed), axis=0)


def _precision_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"fraction": fraction, "k": 0, "precision": None, "recall": None}
    k = max(1, int(math.ceil(len(y_true) * fraction)))
    order = np.argsort(scores)[::-1]
    selected = order[:k]
    positives = int(np.sum(y_true))
    true_positives = int(np.sum(y_true[selected]))
    return {
        "fraction": fraction,
        "k": k,
        "precision": true_positives / k,
        "recall": true_positives / positives if positives else None,
    }


def _metric_payload(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    positives = int(np.sum(y_true))
    negatives = int(len(y_true) - positives)
    payload = {
        "n": int(len(y_true)),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": positives / len(y_true) if len(y_true) else None,
        "average_precision": None,
        "roc_auc": None,
        "precision_at_10_percent": _precision_at_fraction(y_true, scores, 0.10),
        "precision_at_20_percent": _precision_at_fraction(y_true, scores, 0.20),
    }
    if len(y_true) and positives > 0:
        payload["average_precision"] = float(average_precision_score(y_true, scores))
    if positives > 0 and negatives > 0:
        payload["roc_auc"] = float(roc_auc_score(y_true, scores))
        precision, recall, _ = precision_recall_curve(y_true, scores)
        payload["pr_curve_point_count"] = int(len(precision))
        payload["max_precision"] = float(np.max(precision))
        payload["max_recall"] = float(np.max(recall))
    return payload


def evaluate_heuristics(
    feature_rows: list[dict[str, Any]],
    *,
    split: TemporalSplit | None = None,
    target_name: str = PRIMARY_TARGET,
) -> dict[str, Any]:
    """Evaluate cheap heuristic risk scores on a temporal split."""

    eligible_rows = _eligible_rows(feature_rows)
    if split is None:
        split = _default_split(eligible_rows)

    train_rows = [
        row
        for row in eligible_rows
        if row["anchor_trained_task_id"] <= split.train_anchor_task_max
    ]
    test_rows = [
        row
        for row in eligible_rows
        if row["anchor_trained_task_id"] >= split.test_anchor_task_min
    ]
    if not train_rows or not test_rows:
        raise ValueError("temporal split produced empty train or test rows")

    y_test = np.array([int(row[target_name]) for row in test_rows], dtype=int)
    metrics: dict[str, Any] = {}
    for score_name in HEURISTIC_SCORE_NAMES:
        if score_name == "combined_signal":
            scores = _combined_signal_scores(train_rows, test_rows)
        else:
            scores = _score_array(test_rows, score_name)
        metrics[score_name] = _metric_payload(y_test, scores)

    return {
        "target": target_name,
        "temporal_split": {
            "train_anchor_task_max": split.train_anchor_task_max,
            "test_anchor_task_min": split.test_anchor_task_min,
        },
        "train": {
            "n": len(train_rows),
            "positive_count": int(sum(int(row[target_name]) for row in train_rows)),
        },
        "test": {
            "n": len(test_rows),
            "positive_count": int(sum(int(row[target_name]) for row in test_rows)),
        },
        "heuristics": metrics,
    }


def evaluate_logistic_predictor(
    feature_rows: list[dict[str, Any]],
    *,
    split: TemporalSplit | None = None,
    target_name: str = PRIMARY_TARGET,
) -> dict[str, Any]:
    """Fit a small logistic predictor on earlier anchors and test later anchors."""

    eligible_rows = _eligible_rows(feature_rows)
    if split is None:
        split = _default_split(eligible_rows)

    train_rows = [
        row
        for row in eligible_rows
        if row["anchor_trained_task_id"] <= split.train_anchor_task_max
    ]
    test_rows = [
        row
        for row in eligible_rows
        if row["anchor_trained_task_id"] >= split.test_anchor_task_min
    ]
    y_train = np.array([int(row[target_name]) for row in train_rows], dtype=int)
    y_test = np.array([int(row[target_name]) for row in test_rows], dtype=int)

    payload: dict[str, Any] = {
        "target": target_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "train": {
            "n": len(train_rows),
            "positive_count": int(np.sum(y_train)),
        },
        "test": {
            "n": len(test_rows),
            "positive_count": int(np.sum(y_test)),
        },
        "status": "skipped",
        "reason": None,
        "metrics": None,
    }
    if len(train_rows) == 0 or len(test_rows) == 0:
        payload["reason"] = "empty temporal split"
        return payload
    if len(set(y_train.tolist())) < 2:
        payload["reason"] = "training split has only one target class"
        return payload

    x_train = np.array([[row[column] for column in FEATURE_COLUMNS] for row in train_rows])
    x_test = np.array([[row[column] for column in FEATURE_COLUMNS] for row in test_rows])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    payload["status"] = "fit"
    payload["metrics"] = _metric_payload(y_test, scores)
    return payload


def build_predictor_report(
    *,
    signal_payload: dict[str, Any],
    label_payload: dict[str, Any],
    signal_path: str | Path | None = None,
    label_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a complete Task 9 predictor/heuristic report."""

    feature_rows = build_feature_rows(
        signal_payload=signal_payload,
        label_payload=label_payload,
    )
    heuristic_report = evaluate_heuristics(feature_rows)
    split = TemporalSplit(
        train_anchor_task_max=heuristic_report["temporal_split"]["train_anchor_task_max"],
        test_anchor_task_min=heuristic_report["temporal_split"]["test_anchor_task_min"],
    )
    logistic_report = evaluate_logistic_predictor(feature_rows, split=split)

    source_signal = None
    if signal_path is not None:
        signal_path = Path(signal_path)
        source_signal = {"path": str(signal_path), "sha256": sha256_file(signal_path)}
    source_labels = None
    if label_path is not None:
        label_path = Path(label_path)
        source_labels = {"path": str(label_path), "sha256": sha256_file(label_path)}

    return {
        "schema_version": PREDICTOR_REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": PRIMARY_TARGET,
        "source_signal_artifact": source_signal,
        "source_label_artifact": source_labels,
        "definition": {
            "feature_rule": "features use seen_task_eval signal rows with trained_task_id/global_step at or before the label anchor",
            "target_rule": "forgot_any_future from forgetting_labels.json",
            "temporal_evaluation": "earlier anchor tasks are used for scaling/fitting; later anchor tasks are held out",
            "scheduler_use_warning": "this evaluates predictive signal value; scheduler runs must still avoid future labels",
        },
        "feature_summary": {
            "row_count": len(feature_rows),
            "eligible_row_count": len(_eligible_rows(feature_rows)),
            "feature_columns": list(FEATURE_COLUMNS),
            "heuristic_scores": list(HEURISTIC_SCORE_NAMES),
        },
        "heuristic_report": heuristic_report,
        "logistic_report": logistic_report,
        "feature_rows": feature_rows,
    }


def build_predictor_report_from_paths(
    *,
    signal_path: str | Path,
    label_path: str | Path,
) -> dict[str, Any]:
    """Load signal and label artifacts and build a predictor report."""

    signal_path = Path(signal_path)
    label_path = Path(label_path)
    return build_predictor_report(
        signal_payload=_read_json(signal_path),
        label_payload=_read_json(label_path),
        signal_path=signal_path,
        label_path=label_path,
    )


def save_predictor_report(
    *,
    signal_path: str | Path,
    label_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build and save the Task 9 predictor report."""

    report = build_predictor_report_from_paths(
        signal_path=signal_path,
        label_path=label_path,
    )
    _atomic_write_json(Path(output_path), report)
    return report
