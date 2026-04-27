"""Leakage-safe time-to-forgetting target evaluation."""

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

from src.predictors.forgetting_risk import (
    FEATURE_COLUMNS,
    HEURISTIC_SCORE_NAMES,
    TemporalSplit,
    build_feature_rows,
)


TIME_TO_FORGETTING_REPORT_SCHEMA_VERSION = 1
PRIMARY_TIME_TARGET = "first_observed_forgetting_step_delta"


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


def _target_rows_by_anchor(time_payload: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows = time_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("time-to-forgetting payload must contain rows")
    return {
        (
            _as_int(row, "sample_id"),
            _as_int(row, "anchor_trained_task_id"),
            _as_int(row, "anchor_global_step"),
        ): row
        for row in rows
    }


def _label_payload_from_time_targets(time_payload: dict[str, Any]) -> dict[str, Any]:
    rows = time_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("time-to-forgetting payload must contain rows")

    label_rows = []
    for row in rows:
        event_observed = _as_bool(row, "event_observed")
        label_rows.append(
            {
                "sample_id": _as_int(row, "sample_id"),
                "split": str(row["split"]),
                "source_task_id": _as_int(row, "source_task_id"),
                "original_class_id": _as_int(row, "original_class_id"),
                "within_task_label": _as_int(row, "within_task_label"),
                "original_index": _as_int(row, "original_index"),
                "target": _as_int(row, "target"),
                "anchor_trained_task_id": _as_int(row, "anchor_trained_task_id"),
                "anchor_evaluated_task_id": _as_int(row, "anchor_evaluated_task_id"),
                "anchor_global_step": _as_int(row, "anchor_global_step"),
                "anchor_correct": _as_bool(row, "anchor_correct"),
                "anchor_loss": _as_float(row, "anchor_loss"),
                "anchor_confidence": _as_float(row, "anchor_confidence"),
                "anchor_target_probability": _as_float(
                    row,
                    "anchor_target_probability",
                ),
                "anchor_uncertainty": _as_float(row, "anchor_uncertainty"),
                "eligible_for_binary_forgetting": _as_bool(
                    row,
                    "eligible_for_time_to_forgetting",
                ),
                "forgot_any_future": event_observed,
                "forgot_next_eval": event_observed
                and _as_int(row, "first_observed_forgetting_task_delta") == 1,
                "forgot_final_eval": event_observed,
                "label_uses_future_after_task_id": _as_int(
                    row,
                    "anchor_trained_task_id",
                ),
                "leakage_safe": _as_bool(row, "leakage_safe"),
            }
        )
    return {"rows": label_rows}


def build_time_feature_rows(
    *,
    signal_payload: dict[str, Any],
    time_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build timing feature rows using only pre-anchor signal history."""

    base_rows = build_feature_rows(
        signal_payload=signal_payload,
        label_payload=_label_payload_from_time_targets(time_payload),
    )
    targets_by_anchor = _target_rows_by_anchor(time_payload)
    enriched_rows = []
    for row in base_rows:
        key = (
            _as_int(row, "sample_id"),
            _as_int(row, "anchor_trained_task_id"),
            _as_int(row, "anchor_global_step"),
        )
        if key not in targets_by_anchor:
            raise ValueError(f"missing timing target for anchor {key!r}")
        target = targets_by_anchor[key]
        enriched = dict(row)
        enriched.update(
            {
                "eligible_for_time_to_forgetting": _as_bool(
                    target,
                    "eligible_for_time_to_forgetting",
                ),
                "event_observed": _as_bool(target, "event_observed"),
                "right_censored": _as_bool(target, "right_censored"),
                "censoring_type": str(target["censoring_type"]),
                "first_observed_forgetting_task_delta": target.get(
                    "first_observed_forgetting_task_delta"
                ),
                "first_observed_forgetting_step_delta": target.get(
                    "first_observed_forgetting_step_delta"
                ),
                "observed_survival_task_delta": _as_int(
                    target,
                    "observed_survival_task_delta",
                ),
                "observed_survival_step_delta": _as_int(
                    target,
                    "observed_survival_step_delta",
                ),
                "interval_lower_task_delta": target.get("interval_lower_task_delta"),
                "interval_upper_task_delta": target.get("interval_upper_task_delta"),
                "interval_lower_step_delta": target.get("interval_lower_step_delta"),
                "interval_upper_step_delta": target.get("interval_upper_step_delta"),
            }
        )
        enriched_rows.append(enriched)
    return enriched_rows


def _default_split(rows: list[dict[str, Any]]) -> TemporalSplit:
    anchor_tasks = sorted({int(row["anchor_trained_task_id"]) for row in rows})
    if len(anchor_tasks) < 2:
        raise ValueError("temporal timing evaluation requires at least two anchor tasks")
    split_index = max(1, int(len(anchor_tasks) * 0.6)) - 1
    split_index = min(split_index, len(anchor_tasks) - 2)
    return TemporalSplit(
        train_anchor_task_max=anchor_tasks[split_index],
        test_anchor_task_min=anchor_tasks[split_index + 1],
    )


def _eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["eligible_for_time_to_forgetting"]]


def _event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["event_observed"]]


def _score_array(rows: list[dict[str, Any]], score_name: str) -> np.ndarray:
    if score_name == "low_target_probability":
        return np.array([1.0 - row["anchor_target_probability"] for row in rows], dtype=float)
    if score_name == "combined_signal":
        raise ValueError("combined_signal requires train-fitted scaling")
    return np.array([row[score_name] for row in rows], dtype=float)


def _score_arrays(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    score_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if score_name != "combined_signal":
        return _score_array(train_rows, score_name), _score_array(test_rows, score_name)
    keys = (
        "anchor_loss",
        "anchor_uncertainty",
        "low_target_probability",
        "loss_increase_from_previous",
        "target_probability_drop_from_previous",
    )
    train_transformed = []
    test_transformed = []
    for key in keys:
        train_raw = _score_array(train_rows, key)
        test_raw = _score_array(test_rows, key)
        minimum, maximum = _fit_minmax(train_raw)
        train_transformed.append(_normalize(train_raw, minimum, maximum))
        test_transformed.append(_normalize(test_raw, minimum, maximum))
    return (
        np.mean(np.vstack(train_transformed), axis=0),
        np.mean(np.vstack(test_transformed), axis=0),
    )


def _fit_minmax(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    return float(np.min(values)), float(np.max(values))


def _normalize(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if maximum <= minimum:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[middle])
    return int(round((ordered[middle - 1] + ordered[middle]) / 2))


def _concordance(predicted: np.ndarray, observed: np.ndarray) -> float | None:
    concordant = 0
    comparable = 0
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            if observed[left] == observed[right]:
                continue
            comparable += 1
            observed_order = observed[left] < observed[right]
            predicted_order = predicted[left] < predicted[right]
            concordant += int(observed_order == predicted_order)
    if comparable == 0:
        return None
    return concordant / comparable


def _timing_metrics(
    rows: list[dict[str, Any]],
    *,
    predicted_step_deltas: np.ndarray,
    predicted_task_deltas: np.ndarray,
) -> dict[str, Any]:
    event_rows = _event_rows(rows)
    event_indices = [index for index, row in enumerate(rows) if row["event_observed"]]
    censored_indices = [
        index for index, row in enumerate(rows) if row["right_censored"]
    ]
    payload: dict[str, Any] = {
        "n": len(rows),
        "event_observed_count": len(event_rows),
        "right_censored_count": len(censored_indices),
        "mae_step_delta_on_observed_events": None,
        "mae_task_delta_on_observed_events": None,
        "within_one_task_accuracy_on_observed_events": None,
        "timing_concordance_on_observed_events": None,
        "right_censored_predicted_before_last_observation_rate": None,
    }
    if event_indices:
        observed_steps = np.array(
            [rows[index]["first_observed_forgetting_step_delta"] for index in event_indices],
            dtype=float,
        )
        observed_tasks = np.array(
            [rows[index]["first_observed_forgetting_task_delta"] for index in event_indices],
            dtype=float,
        )
        predicted_steps = predicted_step_deltas[event_indices]
        predicted_tasks = predicted_task_deltas[event_indices]
        payload["mae_step_delta_on_observed_events"] = float(
            np.mean(np.abs(predicted_steps - observed_steps))
        )
        payload["mae_task_delta_on_observed_events"] = float(
            np.mean(np.abs(predicted_tasks - observed_tasks))
        )
        payload["within_one_task_accuracy_on_observed_events"] = float(
            np.mean(np.abs(predicted_tasks - observed_tasks) <= 1.0)
        )
        payload["timing_concordance_on_observed_events"] = _concordance(
            predicted_steps,
            observed_steps,
        )
    if censored_indices:
        before_count = 0
        for index in censored_indices:
            before_count += int(
                predicted_step_deltas[index]
                < rows[index]["observed_survival_step_delta"]
            )
        payload["right_censored_predicted_before_last_observation_rate"] = (
            before_count / len(censored_indices)
        )
    return payload


def evaluate_time_to_forgetting_heuristics(
    feature_rows: list[dict[str, Any]],
    *,
    split: TemporalSplit | None = None,
) -> dict[str, Any]:
    """Evaluate simple due-time heuristics on a temporal split."""

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

    train_event_rows = _event_rows(train_rows)
    train_event_step_deltas = [
        int(row["first_observed_forgetting_step_delta"]) for row in train_event_rows
    ]
    train_event_task_deltas = [
        int(row["first_observed_forgetting_task_delta"]) for row in train_event_rows
    ]
    fallback_step_delta = _median_int(
        train_event_step_deltas
        or [int(row["observed_survival_step_delta"]) for row in train_rows]
    )
    fallback_task_delta = _median_int(
        train_event_task_deltas
        or [int(row["observed_survival_task_delta"]) for row in train_rows]
    )
    if fallback_step_delta is None or fallback_task_delta is None:
        raise ValueError("could not derive fallback timing deltas")

    estimators: dict[str, dict[str, Any]] = {}
    constant_step = np.full(len(test_rows), fallback_step_delta, dtype=float)
    constant_task = np.full(len(test_rows), fallback_task_delta, dtype=float)
    estimators["constant_train_median"] = _timing_metrics(
        test_rows,
        predicted_step_deltas=constant_step,
        predicted_task_deltas=constant_task,
    )

    event_step_min = min(train_event_step_deltas) if train_event_step_deltas else fallback_step_delta
    event_step_max = max(train_event_step_deltas) if train_event_step_deltas else fallback_step_delta
    event_task_min = min(train_event_task_deltas) if train_event_task_deltas else fallback_task_delta
    event_task_max = max(train_event_task_deltas) if train_event_task_deltas else fallback_task_delta
    if event_step_max <= event_step_min:
        event_step_max = event_step_min + 1
    if event_task_max <= event_task_min:
        event_task_max = event_task_min + 1

    for score_name in HEURISTIC_SCORE_NAMES:
        train_scores, test_scores = _score_arrays(train_rows, test_rows, score_name)
        score_min, score_max = _fit_minmax(train_scores)
        risk = np.clip(_normalize(test_scores, score_min, score_max), 0.0, 1.0)
        predicted_steps = event_step_max - risk * (event_step_max - event_step_min)
        predicted_tasks = event_task_max - risk * (event_task_max - event_task_min)
        estimators[f"risk_scaled_{score_name}"] = _timing_metrics(
            test_rows,
            predicted_step_deltas=predicted_steps,
            predicted_task_deltas=predicted_tasks,
        )

    return {
        "target": PRIMARY_TIME_TARGET,
        "temporal_split": {
            "train_anchor_task_max": split.train_anchor_task_max,
            "test_anchor_task_min": split.test_anchor_task_min,
        },
        "train": {
            "n": len(train_rows),
            "event_observed_count": len(train_event_rows),
            "right_censored_count": sum(row["right_censored"] for row in train_rows),
        },
        "test": {
            "n": len(test_rows),
            "event_observed_count": len(_event_rows(test_rows)),
            "right_censored_count": sum(row["right_censored"] for row in test_rows),
        },
        "fallback": {
            "median_train_event_step_delta": fallback_step_delta,
            "median_train_event_task_delta": fallback_task_delta,
        },
        "estimators": estimators,
    }


def build_time_to_forgetting_report(
    *,
    signal_payload: dict[str, Any],
    time_payload: dict[str, Any],
    signal_path: str | Path | None = None,
    time_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a complete Task 11 timing-target evaluation report."""

    feature_rows = build_time_feature_rows(
        signal_payload=signal_payload,
        time_payload=time_payload,
    )
    evaluation = evaluate_time_to_forgetting_heuristics(feature_rows)

    source_signal = None
    if signal_path is not None:
        signal_path = Path(signal_path)
        source_signal = {"path": str(signal_path), "sha256": sha256_file(signal_path)}
    source_time = None
    if time_path is not None:
        time_path = Path(time_path)
        source_time = {"path": str(time_path), "sha256": sha256_file(time_path)}

    return {
        "schema_version": TIME_TO_FORGETTING_REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": PRIMARY_TIME_TARGET,
        "source_signal_artifact": source_signal,
        "source_time_to_forgetting_artifact": source_time,
        "definition": {
            "feature_rule": "features use seen_task_eval rows at or before the anchor only",
            "target_rule": "first observed future incorrect evaluation gives an upper-bound time-to-forgetting target",
            "censoring_rule": "anchors that remain correct through the final future evaluation are right-censored",
            "interval_rule": "exact failure time is interval-censored between last observed correct and first observed incorrect evaluation",
        },
        "feature_summary": {
            "row_count": len(feature_rows),
            "eligible_row_count": len(_eligible_rows(feature_rows)),
            "event_observed_count": len(_event_rows(feature_rows)),
            "feature_columns": list(FEATURE_COLUMNS),
            "heuristic_scores": list(HEURISTIC_SCORE_NAMES),
        },
        "evaluation": evaluation,
        "feature_rows": feature_rows,
    }


def build_time_to_forgetting_report_from_paths(
    *,
    signal_path: str | Path,
    time_path: str | Path,
) -> dict[str, Any]:
    signal_path = Path(signal_path)
    time_path = Path(time_path)
    return build_time_to_forgetting_report(
        signal_payload=_read_json(signal_path),
        time_payload=_read_json(time_path),
        signal_path=signal_path,
        time_path=time_path,
    )


def save_time_to_forgetting_report(
    *,
    signal_path: str | Path,
    time_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = build_time_to_forgetting_report_from_paths(
        signal_path=signal_path,
        time_path=time_path,
    )
    _atomic_write_json(Path(output_path), report)
    return report
