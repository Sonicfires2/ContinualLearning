"""Forgetting-risk prediction from NLP continual-learning eval traces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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


NLP_FORGETTING_REPORT_SCHEMA_VERSION = 1
NLP_FORGETTING_TARGET = "forgot_any_future"
NLP_FEATURE_COLUMNS = (
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
NLP_HEURISTIC_SCORE_NAMES = (
    "anchor_loss",
    "anchor_uncertainty",
    "low_target_probability",
    "loss_increase_from_previous",
    "target_probability_drop_from_previous",
    "combined_signal",
)


@dataclass(frozen=True)
class NLPTemporalSplit:
    """Temporal train/test split over anchor task IDs."""

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
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _signal_rows(signal_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = signal_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("NLP signal payload must contain a rows list")
    if not rows:
        raise ValueError("NLP signal payload contains no rows")
    return rows


def _group_signal_history(signal_payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in _signal_rows(signal_payload):
        grouped.setdefault(_as_int(row, "sample_id"), []).append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                _as_int(row, "trained_task_id"),
                _as_int(row, "global_step"),
            )
        )
    return grouped


def _uncertainty(row: dict[str, Any]) -> float:
    if "uncertainty" in row:
        return _as_float(row, "uncertainty")
    return 1.0 - _as_float(row, "confidence")


def _feature_row(
    *,
    anchor: dict[str, Any],
    previous: dict[str, Any] | None,
    history: list[dict[str, Any]],
    future_rows: list[dict[str, Any]],
    max_anchor_task: int,
) -> dict[str, Any]:
    anchor_loss = _as_float(anchor, "loss")
    anchor_confidence = _as_float(anchor, "confidence")
    anchor_uncertainty = _uncertainty(anchor)
    anchor_target_probability = _as_float(anchor, "target_probability")
    previous_loss = _as_float(previous, "loss") if previous is not None else anchor_loss
    previous_uncertainty = _uncertainty(previous) if previous is not None else anchor_uncertainty
    previous_target_probability = (
        _as_float(previous, "target_probability")
        if previous is not None
        else anchor_target_probability
    )
    loss_delta = anchor_loss - previous_loss
    target_probability_delta = anchor_target_probability - previous_target_probability
    anchor_task = _as_int(anchor, "trained_task_id")
    source_task = _as_int(anchor, "task_id")
    anchor_correct = _as_bool(anchor, "correct")
    future_correct = [_as_bool(row, "correct") for row in future_rows]
    history_correct = [_as_bool(row, "correct") for row in history]
    history_losses = [_as_float(row, "loss") for row in history]
    history_target_probabilities = [_as_float(row, "target_probability") for row in history]
    history_uncertainties = [_uncertainty(row) for row in history]

    return {
        "sample_id": _as_int(anchor, "sample_id"),
        "split": "eval",
        "source_task_id": source_task,
        "eval_task_id": _as_int(anchor, "eval_task_id"),
        "original_class_id": _as_int(anchor, "original_class_id"),
        "label": _as_int(anchor, "label"),
        "anchor_trained_task_id": anchor_task,
        "anchor_global_step": _as_int(anchor, "global_step"),
        "anchor_correct": anchor_correct,
        "eligible_for_binary_forgetting": anchor_correct and bool(future_rows),
        "future_eval_count": len(future_rows),
        "forgot_next_eval": bool(future_rows and not _as_bool(future_rows[0], "correct")),
        "forgot_final_eval": bool(future_rows and not _as_bool(future_rows[-1], "correct")),
        "forgot_any_future": bool(any(not value for value in future_correct)),
        "future_correct_count": int(sum(future_correct)),
        "anchor_loss": anchor_loss,
        "anchor_uncertainty": anchor_uncertainty,
        "anchor_confidence": anchor_confidence,
        "anchor_target_probability": anchor_target_probability,
        "history_eval_count": len(history),
        "has_previous_eval": int(previous is not None),
        "previous_correct": (
            int(_as_bool(previous, "correct")) if previous is not None else int(anchor_correct)
        ),
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
        "correct_history_rate": sum(history_correct) / len(history_correct),
        "tasks_since_source": anchor_task - source_task,
        "anchor_task_progress": (
            anchor_task / max_anchor_task if max_anchor_task > 0 else 0.0
        ),
        "feature_uses_rows_up_to_trained_task_id": anchor_task,
        "feature_uses_rows_up_to_global_step": _as_int(anchor, "global_step"),
        "leakage_safe": True,
    }


def build_nlp_forgetting_feature_rows(signal_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build leakage-safe anchor rows and future-forgetting labels for NLP runs."""

    grouped = _group_signal_history(signal_payload)
    all_anchor_tasks = [
        _as_int(row, "trained_task_id")
        for rows in grouped.values()
        for row in rows[:-1]
    ]
    if not all_anchor_tasks:
        raise ValueError("NLP signal payload has no anchors with future evaluations")
    max_anchor_task = max(all_anchor_tasks)
    feature_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        for index, anchor in enumerate(rows[:-1]):
            history = rows[: index + 1]
            future_rows = rows[index + 1 :]
            previous = rows[index - 1] if index > 0 else None
            feature_rows.append(
                _feature_row(
                    anchor=anchor,
                    previous=previous,
                    history=history,
                    future_rows=future_rows,
                    max_anchor_task=max_anchor_task,
                )
            )
    if not all(row["leakage_safe"] for row in feature_rows):
        raise ValueError("NLP feature builder produced a non-leakage-safe row")
    return feature_rows


def _eligible_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in feature_rows if row["eligible_for_binary_forgetting"]]


def default_nlp_temporal_split(feature_rows: list[dict[str, Any]]) -> NLPTemporalSplit:
    eligible = _eligible_rows(feature_rows)
    anchor_tasks = sorted({int(row["anchor_trained_task_id"]) for row in eligible})
    if len(anchor_tasks) < 2:
        raise ValueError("temporal evaluation requires at least two anchor tasks")
    split_index = max(1, int(len(anchor_tasks) * 0.6)) - 1
    split_index = min(split_index, len(anchor_tasks) - 2)
    return NLPTemporalSplit(
        train_anchor_task_max=anchor_tasks[split_index],
        test_anchor_task_min=anchor_tasks[split_index + 1],
    )


def _score_array(rows: list[dict[str, Any]], score_name: str) -> np.ndarray:
    if score_name == "low_target_probability":
        return np.asarray(
            [1.0 - row["anchor_target_probability"] for row in rows],
            dtype=float,
        )
    if score_name == "combined_signal":
        raise ValueError("combined_signal requires fitted min-max scaling")
    return np.asarray([row[score_name] for row in rows], dtype=float)


def _fit_minmax(train_rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    fitted = {}
    for key in keys:
        values = _score_array(train_rows, key)
        fitted[key] = (float(np.min(values)), float(np.max(values)))
    return fitted


def _minmax(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
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
        transformed.append(_minmax(_score_array(test_rows, key), *fitted[key]))
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


def _split_rows(
    feature_rows: list[dict[str, Any]],
    split: NLPTemporalSplit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = _eligible_rows(feature_rows)
    train_rows = [
        row
        for row in eligible
        if row["anchor_trained_task_id"] <= split.train_anchor_task_max
    ]
    test_rows = [
        row
        for row in eligible
        if row["anchor_trained_task_id"] >= split.test_anchor_task_min
    ]
    if not train_rows or not test_rows:
        raise ValueError("temporal split produced empty train or test rows")
    return train_rows, test_rows


def evaluate_nlp_heuristics(
    feature_rows: list[dict[str, Any]],
    *,
    split: NLPTemporalSplit | None = None,
    target_name: str = NLP_FORGETTING_TARGET,
) -> dict[str, Any]:
    """Evaluate cheap NLP forgetting-risk heuristic scores."""

    split = split or default_nlp_temporal_split(feature_rows)
    train_rows, test_rows = _split_rows(feature_rows, split)
    y_test = np.asarray([int(row[target_name]) for row in test_rows], dtype=int)
    heuristic_metrics: dict[str, Any] = {}
    for score_name in NLP_HEURISTIC_SCORE_NAMES:
        scores = (
            _combined_signal_scores(train_rows, test_rows)
            if score_name == "combined_signal"
            else _score_array(test_rows, score_name)
        )
        heuristic_metrics[score_name] = _metric_payload(y_test, scores)
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
        "heuristics": heuristic_metrics,
    }


def evaluate_nlp_logistic_predictor(
    feature_rows: list[dict[str, Any]],
    *,
    split: NLPTemporalSplit | None = None,
    target_name: str = NLP_FORGETTING_TARGET,
) -> dict[str, Any]:
    """Train logistic regression on earlier anchors and test later anchors."""

    split = split or default_nlp_temporal_split(feature_rows)
    train_rows, test_rows = _split_rows(feature_rows, split)
    y_train = np.asarray([int(row[target_name]) for row in train_rows], dtype=int)
    y_test = np.asarray([int(row[target_name]) for row in test_rows], dtype=int)
    payload: dict[str, Any] = {
        "target": target_name,
        "feature_columns": list(NLP_FEATURE_COLUMNS),
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
    if len(set(y_train.tolist())) < 2:
        payload["reason"] = "training split has only one target class"
        return payload
    x_train = np.asarray([[row[column] for column in NLP_FEATURE_COLUMNS] for row in train_rows])
    x_test = np.asarray([[row[column] for column in NLP_FEATURE_COLUMNS] for row in test_rows])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    payload["status"] = "fit"
    payload["metrics"] = _metric_payload(y_test, scores)
    return payload


def build_nlp_forgetting_report(signal_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a full leakage-safe NLP forgetting-predictor report."""

    feature_rows = build_nlp_forgetting_feature_rows(signal_payload)
    split = default_nlp_temporal_split(feature_rows)
    heuristic_report = evaluate_nlp_heuristics(feature_rows, split=split)
    logistic_report = evaluate_nlp_logistic_predictor(feature_rows, split=split)
    eligible = _eligible_rows(feature_rows)
    return {
        "schema_version": NLP_FORGETTING_REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": NLP_FORGETTING_TARGET,
        "definition": {
            "anchor": "one eval row for a text example after a trained task",
            "features": "current and past eval rows for the same sample only",
            "label": "whether any later eval row for that sample is incorrect",
            "temporal_evaluation": "earlier anchor tasks train/scale, later anchor tasks test",
            "leakage_guard": "future eval rows are used only for labels, never features",
        },
        "feature_summary": {
            "row_count": len(feature_rows),
            "eligible_row_count": len(eligible),
            "positive_count": int(sum(int(row[NLP_FORGETTING_TARGET]) for row in eligible)),
            "feature_columns": list(NLP_FEATURE_COLUMNS),
            "heuristic_scores": list(NLP_HEURISTIC_SCORE_NAMES),
        },
        "heuristic_report": heuristic_report,
        "logistic_report": logistic_report,
        "feature_rows": feature_rows,
    }


def build_nlp_forgetting_report_from_path(signal_path: str | Path) -> dict[str, Any]:
    """Load an NLP eval-signals artifact and build a predictor report."""

    return build_nlp_forgetting_report(_read_json(Path(signal_path)))


def save_nlp_forgetting_report(
    *,
    signal_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build and save an NLP forgetting-predictor report."""

    report = build_nlp_forgetting_report_from_path(signal_path)
    _atomic_write_json(Path(output_path), report)
    return report

