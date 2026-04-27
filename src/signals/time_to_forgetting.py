"""Time-to-forgetting target construction from sample signal logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable


TIME_TO_FORGETTING_SCHEMA_VERSION = 1
PRIMARY_TIME_TARGET = "first_observed_forgetting_step_delta"


@dataclass(frozen=True)
class TimeToForgettingRow:
    """Time-to-event target anchored at one retained evaluation observation."""

    sample_id: int
    split: str
    source_task_id: int
    original_class_id: int
    within_task_label: int
    original_index: int
    target: int
    anchor_trained_task_id: int
    anchor_evaluated_task_id: int
    anchor_global_step: int
    anchor_correct: bool
    anchor_loss: float
    anchor_confidence: float
    anchor_target_probability: float
    anchor_uncertainty: float
    eligible_for_time_to_forgetting: bool
    future_eval_count: int
    event_observed: bool
    censoring_type: str
    first_forgetting_trained_task_id: int | None
    first_forgetting_global_step: int | None
    first_observed_forgetting_task_delta: int | None
    first_observed_forgetting_step_delta: int | None
    last_observed_correct_trained_task_id: int | None
    last_observed_correct_global_step: int | None
    interval_lower_task_delta: int | None
    interval_lower_step_delta: int | None
    interval_upper_task_delta: int | None
    interval_upper_step_delta: int | None
    survived_until_trained_task_id: int
    survived_until_global_step: int
    observed_survival_task_delta: int
    observed_survival_step_delta: int
    interval_censored: bool
    right_censored: bool
    leakage_safe: bool


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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


def _finite_float(value: Any, *, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _required_int(row: dict[str, Any], key: str) -> int:
    if key not in row:
        raise KeyError(f"signal row is missing required key {key!r}")
    return int(row[key])


def _required_float(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise KeyError(f"signal row is missing required key {key!r}")
    return _finite_float(row[key], field_name=key)


def _required_bool(row: dict[str, Any], key: str) -> bool:
    if key not in row:
        raise KeyError(f"signal row is missing required key {key!r}")
    return bool(row[key])


def _seen_eval_rows(signal_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = signal_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("signal payload must contain a list field named 'rows'")

    eval_rows = [
        row for row in rows if row.get("observation_type") == "seen_task_eval"
    ]
    if not eval_rows:
        raise ValueError("signal payload contains no seen_task_eval rows")
    return eval_rows


def _group_by_sample(rows: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_required_int(row, "sample_id"), []).append(row)

    for sample_rows in grouped.values():
        sample_rows.sort(
            key=lambda row: (
                _required_int(row, "trained_task_id"),
                _required_int(row, "global_step"),
            )
        )
    return grouped


def _validate_eval_row(row: dict[str, Any]) -> None:
    trained_task_id = _required_int(row, "trained_task_id")
    evaluated_task_id = _required_int(row, "evaluated_task_id")
    source_task_id = _required_int(row, "source_task_id")
    if evaluated_task_id != source_task_id:
        raise ValueError(
            "seen_task_eval row evaluated_task_id must match source_task_id for "
            f"sample {row.get('sample_id')!r}"
        )
    if trained_task_id < source_task_id:
        raise ValueError(
            "seen_task_eval row appears before its source task was trained for "
            f"sample {row.get('sample_id')!r}"
        )


def _time_row(anchor: dict[str, Any], future_rows: list[dict[str, Any]]) -> TimeToForgettingRow:
    if not future_rows:
        raise ValueError("future_rows must not be empty")

    anchor_task = _required_int(anchor, "trained_task_id")
    anchor_step = _required_int(anchor, "global_step")
    for future_row in future_rows:
        if _required_int(future_row, "trained_task_id") <= anchor_task:
            raise ValueError("future rows must occur strictly after the anchor row")

    anchor_correct = _required_bool(anchor, "correct")
    final_row = future_rows[-1]
    final_task = _required_int(final_row, "trained_task_id")
    final_step = _required_int(final_row, "global_step")

    event_row = None
    if anchor_correct:
        for future_row in future_rows:
            if not _required_bool(future_row, "correct"):
                event_row = future_row
                break

    if not anchor_correct:
        censoring_type = "not_retained_at_anchor"
        event_observed = False
        right_censored = False
        interval_censored = False
        first_task = None
        first_step = None
        upper_task_delta = None
        upper_step_delta = None
        lower_task_delta = None
        lower_step_delta = None
        last_correct_task = None
        last_correct_step = None
        survival_task_delta = 0
        survival_step_delta = 0
        survived_until_task = anchor_task
        survived_until_step = anchor_step
    elif event_row is None:
        censoring_type = "right_censored"
        event_observed = False
        right_censored = True
        interval_censored = False
        first_task = None
        first_step = None
        upper_task_delta = None
        upper_step_delta = None
        lower_task_delta = final_task - anchor_task
        lower_step_delta = final_step - anchor_step
        last_correct = future_rows[-1]
        last_correct_task = _required_int(last_correct, "trained_task_id")
        last_correct_step = _required_int(last_correct, "global_step")
        survival_task_delta = final_task - anchor_task
        survival_step_delta = final_step - anchor_step
        survived_until_task = final_task
        survived_until_step = final_step
    else:
        censoring_type = "event_observed_interval_censored"
        event_observed = True
        right_censored = False
        interval_censored = True
        first_task = _required_int(event_row, "trained_task_id")
        first_step = _required_int(event_row, "global_step")
        event_index = future_rows.index(event_row)
        previous_rows = [anchor, *future_rows[:event_index]]
        correct_previous_rows = [
            row for row in previous_rows if _required_bool(row, "correct")
        ]
        last_correct = correct_previous_rows[-1]
        last_correct_task = _required_int(last_correct, "trained_task_id")
        last_correct_step = _required_int(last_correct, "global_step")
        lower_task_delta = last_correct_task - anchor_task
        lower_step_delta = last_correct_step - anchor_step
        upper_task_delta = first_task - anchor_task
        upper_step_delta = first_step - anchor_step
        survival_task_delta = lower_task_delta
        survival_step_delta = lower_step_delta
        survived_until_task = last_correct_task
        survived_until_step = last_correct_step

    return TimeToForgettingRow(
        sample_id=_required_int(anchor, "sample_id"),
        split=str(anchor["split"]),
        source_task_id=_required_int(anchor, "source_task_id"),
        original_class_id=_required_int(anchor, "original_class_id"),
        within_task_label=_required_int(anchor, "within_task_label"),
        original_index=_required_int(anchor, "original_index"),
        target=_required_int(anchor, "target"),
        anchor_trained_task_id=anchor_task,
        anchor_evaluated_task_id=_required_int(anchor, "evaluated_task_id"),
        anchor_global_step=anchor_step,
        anchor_correct=anchor_correct,
        anchor_loss=_required_float(anchor, "loss"),
        anchor_confidence=_required_float(anchor, "confidence"),
        anchor_target_probability=_required_float(anchor, "target_probability"),
        anchor_uncertainty=_required_float(anchor, "uncertainty"),
        eligible_for_time_to_forgetting=anchor_correct,
        future_eval_count=len(future_rows),
        event_observed=event_observed,
        censoring_type=censoring_type,
        first_forgetting_trained_task_id=first_task,
        first_forgetting_global_step=first_step,
        first_observed_forgetting_task_delta=(
            upper_task_delta if event_observed else None
        ),
        first_observed_forgetting_step_delta=(
            upper_step_delta if event_observed else None
        ),
        last_observed_correct_trained_task_id=last_correct_task,
        last_observed_correct_global_step=last_correct_step,
        interval_lower_task_delta=lower_task_delta,
        interval_lower_step_delta=lower_step_delta,
        interval_upper_task_delta=upper_task_delta,
        interval_upper_step_delta=upper_step_delta,
        survived_until_trained_task_id=survived_until_task,
        survived_until_global_step=survived_until_step,
        observed_survival_task_delta=survival_task_delta,
        observed_survival_step_delta=survival_step_delta,
        interval_censored=interval_censored,
        right_censored=right_censored,
        leakage_safe=all(
            _required_int(row, "trained_task_id") > anchor_task for row in future_rows
        ),
    )


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def summarize_time_to_forgetting(rows: list[TimeToForgettingRow]) -> dict[str, Any]:
    """Return target counts and timing summaries."""

    anchor_count = len(rows)
    eligible_count = sum(row.eligible_for_time_to_forgetting for row in rows)
    observed_events = [row for row in rows if row.event_observed]
    right_censored = [row for row in rows if row.right_censored]
    not_retained = [
        row for row in rows if row.censoring_type == "not_retained_at_anchor"
    ]
    event_task_deltas = [
        int(row.first_observed_forgetting_task_delta)
        for row in observed_events
        if row.first_observed_forgetting_task_delta is not None
    ]
    event_step_deltas = [
        int(row.first_observed_forgetting_step_delta)
        for row in observed_events
        if row.first_observed_forgetting_step_delta is not None
    ]

    return {
        "schema_version": TIME_TO_FORGETTING_SCHEMA_VERSION,
        "primary_time_target": PRIMARY_TIME_TARGET,
        "anchor_count": anchor_count,
        "eligible_for_time_to_forgetting_count": eligible_count,
        "event_observed_count": len(observed_events),
        "right_censored_count": len(right_censored),
        "not_retained_at_anchor_count": len(not_retained),
        "event_rate_over_eligible": (
            len(observed_events) / eligible_count if eligible_count else None
        ),
        "median_observed_forgetting_task_delta": _median(event_task_deltas),
        "median_observed_forgetting_step_delta": _median(event_step_deltas),
        "min_observed_forgetting_task_delta": (
            min(event_task_deltas) if event_task_deltas else None
        ),
        "max_observed_forgetting_task_delta": (
            max(event_task_deltas) if event_task_deltas else None
        ),
        "min_observed_forgetting_step_delta": (
            min(event_step_deltas) if event_step_deltas else None
        ),
        "max_observed_forgetting_step_delta": (
            max(event_step_deltas) if event_step_deltas else None
        ),
    }


def build_time_to_forgetting_artifact(
    signal_payload: dict[str, Any],
    *,
    signal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create an auditable time-to-forgetting target artifact."""

    eval_rows = _seen_eval_rows(signal_payload)
    for row in eval_rows:
        _validate_eval_row(row)

    target_rows: list[TimeToForgettingRow] = []
    for sample_rows in _group_by_sample(eval_rows).values():
        for index, anchor in enumerate(sample_rows[:-1]):
            target_rows.append(_time_row(anchor, sample_rows[index + 1 :]))

    if not all(row.leakage_safe for row in target_rows):
        raise ValueError("at least one target row uses non-future evaluation data")

    source = None
    if signal_path is not None:
        signal_path = Path(signal_path)
        source = {
            "path": str(signal_path),
            "sha256": sha256_file(signal_path),
        }

    return {
        "schema_version": TIME_TO_FORGETTING_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "primary_time_target": PRIMARY_TIME_TARGET,
        "definition": {
            "anchor_rows": "seen_task_eval rows with at least one later seen_task_eval row for the same sample_id",
            "eligible_for_time_to_forgetting": "anchor_correct == true",
            "event_observed": "anchor_correct == true and at least one later evaluation for the same sample_id is incorrect",
            "first_observed_forgetting_step_delta": "first later incorrect evaluation global_step minus anchor_global_step",
            "first_observed_forgetting_task_delta": "first later incorrect trained_task_id minus anchor_trained_task_id",
            "interval_censoring": "exact failure occurs after the last observed correct evaluation and at or before the first observed incorrect evaluation",
            "right_censoring": "the sample remains correct through the last available future evaluation",
            "not_retained_at_anchor": "the anchor was already incorrect, so time-to-forgetting is undefined",
            "scheduler_warning": "online schedulers may use these targets for offline validation only; future rows must never be used during scheduling",
        },
        "source_signal_artifact": source,
        "source_signal_summary": signal_payload.get("summary", {}),
        "summary": summarize_time_to_forgetting(target_rows),
        "rows": [asdict(row) for row in target_rows],
    }


def build_time_to_forgetting_artifact_from_path(signal_path: str | Path) -> dict[str, Any]:
    path = Path(signal_path)
    return build_time_to_forgetting_artifact(_read_json(path), signal_path=path)


def save_time_to_forgetting_artifact(
    *,
    signal_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    artifact = build_time_to_forgetting_artifact_from_path(signal_path)
    _atomic_write_json(Path(output_path), artifact)
    return artifact
