"""Derive sample-level future forgetting labels from signal logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable


FORGETTING_LABEL_SCHEMA_VERSION = 1
PRIMARY_FORGETTING_LABEL = "forgot_any_future"


@dataclass(frozen=True)
class ForgettingLabelRow:
    """Prediction target anchored at one seen-task evaluation observation."""

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
    eligible_for_binary_forgetting: bool
    future_eval_count: int
    next_trained_task_id: int | None
    final_trained_task_id: int
    forgot_next_eval: bool
    forgot_final_eval: bool
    forgot_any_future: bool
    next_loss_delta: float | None
    final_loss_delta: float
    max_future_loss_increase: float
    next_target_probability_drop: float | None
    final_target_probability_drop: float
    max_future_target_probability_drop: float
    next_confidence_drop: float | None
    final_confidence_drop: float
    max_future_confidence_drop: float
    future_min_target_probability: float
    future_max_loss: float
    label_uses_future_after_task_id: int
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
    """Return the SHA-256 digest of a file."""

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


def _label_row(anchor: dict[str, Any], future_rows: list[dict[str, Any]]) -> ForgettingLabelRow:
    if not future_rows:
        raise ValueError("future_rows must not be empty")

    for future_row in future_rows:
        if _required_int(future_row, "trained_task_id") <= _required_int(
            anchor,
            "trained_task_id",
        ):
            raise ValueError("future rows must occur strictly after the anchor row")

    next_row = future_rows[0]
    final_row = future_rows[-1]
    anchor_correct = _required_bool(anchor, "correct")
    future_correct_values = [_required_bool(row, "correct") for row in future_rows]
    anchor_loss = _required_float(anchor, "loss")
    anchor_target_probability = _required_float(anchor, "target_probability")
    anchor_confidence = _required_float(anchor, "confidence")
    future_losses = [_required_float(row, "loss") for row in future_rows]
    future_target_probabilities = [
        _required_float(row, "target_probability") for row in future_rows
    ]
    future_confidences = [_required_float(row, "confidence") for row in future_rows]

    return ForgettingLabelRow(
        sample_id=_required_int(anchor, "sample_id"),
        split=str(anchor["split"]),
        source_task_id=_required_int(anchor, "source_task_id"),
        original_class_id=_required_int(anchor, "original_class_id"),
        within_task_label=_required_int(anchor, "within_task_label"),
        original_index=_required_int(anchor, "original_index"),
        target=_required_int(anchor, "target"),
        anchor_trained_task_id=_required_int(anchor, "trained_task_id"),
        anchor_evaluated_task_id=_required_int(anchor, "evaluated_task_id"),
        anchor_global_step=_required_int(anchor, "global_step"),
        anchor_correct=anchor_correct,
        anchor_loss=anchor_loss,
        anchor_confidence=anchor_confidence,
        anchor_target_probability=anchor_target_probability,
        anchor_uncertainty=_required_float(anchor, "uncertainty"),
        eligible_for_binary_forgetting=anchor_correct,
        future_eval_count=len(future_rows),
        next_trained_task_id=_required_int(next_row, "trained_task_id"),
        final_trained_task_id=_required_int(final_row, "trained_task_id"),
        forgot_next_eval=anchor_correct and not _required_bool(next_row, "correct"),
        forgot_final_eval=anchor_correct and not _required_bool(final_row, "correct"),
        forgot_any_future=anchor_correct and any(not value for value in future_correct_values),
        next_loss_delta=_required_float(next_row, "loss") - anchor_loss,
        final_loss_delta=_required_float(final_row, "loss") - anchor_loss,
        max_future_loss_increase=max(loss - anchor_loss for loss in future_losses),
        next_target_probability_drop=anchor_target_probability
        - _required_float(next_row, "target_probability"),
        final_target_probability_drop=anchor_target_probability
        - _required_float(final_row, "target_probability"),
        max_future_target_probability_drop=max(
            anchor_target_probability - probability
            for probability in future_target_probabilities
        ),
        next_confidence_drop=anchor_confidence - _required_float(next_row, "confidence"),
        final_confidence_drop=anchor_confidence - _required_float(final_row, "confidence"),
        max_future_confidence_drop=max(
            anchor_confidence - confidence for confidence in future_confidences
        ),
        future_min_target_probability=min(future_target_probabilities),
        future_max_loss=max(future_losses),
        label_uses_future_after_task_id=_required_int(anchor, "trained_task_id"),
        leakage_safe=all(
            _required_int(row, "trained_task_id")
            > _required_int(anchor, "trained_task_id")
            for row in future_rows
        ),
    )


def _increment_nested_count(
    counts: dict[str, dict[str, int]],
    group_name: str,
    group_value: Any,
    *,
    eligible: bool,
    forgot: bool,
) -> None:
    key = str(group_value)
    group = counts.setdefault(
        group_name,
        {},
    )
    group[key] = group.get(key, 0) + 1

    eligible_group = counts.setdefault(f"{group_name}_eligible", {})
    positive_group = counts.setdefault(f"{group_name}_forgot_any_future", {})
    eligible_group[key] = eligible_group.get(key, 0) + int(eligible)
    positive_group[key] = positive_group.get(key, 0) + int(forgot)


def summarize_forgetting_labels(rows: list[ForgettingLabelRow]) -> dict[str, Any]:
    """Return label counts and rates for sanity checking."""

    anchor_count = len(rows)
    eligible_count = sum(row.eligible_for_binary_forgetting for row in rows)
    forgot_next = sum(row.forgot_next_eval for row in rows)
    forgot_final = sum(row.forgot_final_eval for row in rows)
    forgot_any = sum(row.forgot_any_future for row in rows)
    counts: dict[str, dict[str, int]] = {}

    for row in rows:
        _increment_nested_count(
            counts,
            "by_anchor_task",
            row.anchor_trained_task_id,
            eligible=row.eligible_for_binary_forgetting,
            forgot=row.forgot_any_future,
        )
        _increment_nested_count(
            counts,
            "by_source_task",
            row.source_task_id,
            eligible=row.eligible_for_binary_forgetting,
            forgot=row.forgot_any_future,
        )
        _increment_nested_count(
            counts,
            "by_original_class",
            row.original_class_id,
            eligible=row.eligible_for_binary_forgetting,
            forgot=row.forgot_any_future,
        )
        _increment_nested_count(
            counts,
            "by_split",
            row.split,
            eligible=row.eligible_for_binary_forgetting,
            forgot=row.forgot_any_future,
        )

    return {
        "schema_version": FORGETTING_LABEL_SCHEMA_VERSION,
        "primary_label": PRIMARY_FORGETTING_LABEL,
        "anchor_count": anchor_count,
        "eligible_for_binary_forgetting_count": eligible_count,
        "forgot_next_eval_count": forgot_next,
        "forgot_final_eval_count": forgot_final,
        "forgot_any_future_count": forgot_any,
        "forgot_any_future_rate_over_eligible": (
            forgot_any / eligible_count if eligible_count else None
        ),
        "forgot_next_eval_rate_over_eligible": (
            forgot_next / eligible_count if eligible_count else None
        ),
        "forgot_final_eval_rate_over_eligible": (
            forgot_final / eligible_count if eligible_count else None
        ),
        "counts": counts,
    }


def build_forgetting_label_artifact(
    signal_payload: dict[str, Any],
    *,
    signal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create an auditable future-forgetting label artifact from signal rows."""

    eval_rows = _seen_eval_rows(signal_payload)
    for row in eval_rows:
        _validate_eval_row(row)

    label_rows: list[ForgettingLabelRow] = []
    for sample_rows in _group_by_sample(eval_rows).values():
        for index, anchor in enumerate(sample_rows[:-1]):
            future_rows = sample_rows[index + 1 :]
            label_rows.append(_label_row(anchor, future_rows))

    if not all(row.leakage_safe for row in label_rows):
        raise ValueError("at least one label row uses non-future evaluation data")

    signal_summary = signal_payload.get("summary", {})
    source = None
    if signal_path is not None:
        signal_path = Path(signal_path)
        source = {
            "path": str(signal_path),
            "sha256": sha256_file(signal_path),
        }

    return {
        "schema_version": FORGETTING_LABEL_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "primary_label": PRIMARY_FORGETTING_LABEL,
        "definition": {
            "anchor_rows": "seen_task_eval rows with at least one later seen_task_eval row for the same sample_id",
            "eligible_for_binary_forgetting": "anchor_correct == true",
            "forgot_any_future": "anchor_correct == true and at least one later evaluation for the same sample_id is incorrect",
            "forgot_next_eval": "anchor_correct == true and the next later evaluation for the same sample_id is incorrect",
            "forgot_final_eval": "anchor_correct == true and the final available evaluation for the same sample_id is incorrect",
            "continuous_targets": [
                "max_future_loss_increase",
                "final_loss_delta",
                "max_future_target_probability_drop",
                "final_target_probability_drop",
                "max_future_confidence_drop",
                "final_confidence_drop",
            ],
            "leakage_guard": "future labels only use rows with trained_task_id greater than anchor_trained_task_id",
        },
        "source_signal_artifact": source,
        "source_signal_summary": signal_summary,
        "summary": summarize_forgetting_labels(label_rows),
        "rows": [asdict(row) for row in label_rows],
    }


def build_forgetting_label_artifact_from_path(signal_path: str | Path) -> dict[str, Any]:
    """Load `sample_signals.json` and derive forgetting labels."""

    path = Path(signal_path)
    return build_forgetting_label_artifact(_read_json(path), signal_path=path)


def save_forgetting_label_artifact(
    *,
    signal_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build and save a forgetting-label artifact."""

    artifact = build_forgetting_label_artifact_from_path(signal_path)
    _atomic_write_json(Path(output_path), artifact)
    return artifact
