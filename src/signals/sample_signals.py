"""Sample-level signal logging for forgetting-risk analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import torch
from torch.nn import functional as F


SIGNAL_SCHEMA_VERSION = 1
SIGNAL_FIELDS = (
    "sample_id",
    "source_task_id",
    "original_class_id",
    "within_task_label",
    "original_index",
    "split",
    "target",
    "observation_type",
    "trained_task_id",
    "evaluated_task_id",
    "epoch",
    "global_step",
    "is_replay",
    "replay_count",
    "last_replay_step",
    "loss",
    "predicted_class",
    "correct",
    "confidence",
    "target_probability",
    "uncertainty",
    "entropy",
)


@dataclass(frozen=True)
class SampleSignalRow:
    """One model observation for one sample at one training or evaluation point."""

    sample_id: int
    source_task_id: int
    original_class_id: int
    within_task_label: int
    original_index: int
    split: str
    target: int
    observation_type: str
    trained_task_id: int
    evaluated_task_id: int | None
    epoch: int | None
    global_step: int
    is_replay: bool
    replay_count: int
    last_replay_step: int | None
    loss: float
    predicted_class: int
    correct: bool
    confidence: float
    target_probability: float
    uncertainty: float
    entropy: float


def _as_sequence(value: Any, *, field_name: str, expected_len: int) -> Sequence[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            values = [value.item()]
        else:
            values = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]

    if len(values) != expected_len:
        raise ValueError(
            f"signal field {field_name!r} has length {len(values)}, expected {expected_len}"
        )
    return values


def _int_field(batch: dict[str, Any], key: str, index: int, batch_size: int) -> int:
    if key not in batch:
        raise KeyError(f"signal logging requires batch metadata key {key!r}")
    values = _as_sequence(batch[key], field_name=key, expected_len=batch_size)
    return int(values[index])


def _optional_int_field(
    batch: dict[str, Any],
    key: str,
    index: int,
    batch_size: int,
    *,
    default: int | None,
) -> int | None:
    if key not in batch:
        return default
    value = _as_sequence(batch[key], field_name=key, expected_len=batch_size)[index]
    if value is None:
        return None
    return int(value)


def _bool_field(
    batch: dict[str, Any],
    key: str,
    index: int,
    batch_size: int,
    *,
    default: bool,
) -> bool:
    if key not in batch:
        return default
    value = _as_sequence(batch[key], field_name=key, expected_len=batch_size)[index]
    return bool(value)


def _str_field(batch: dict[str, Any], key: str, index: int, batch_size: int) -> str:
    if key not in batch:
        raise KeyError(f"signal logging requires batch metadata key {key!r}")
    values = _as_sequence(batch[key], field_name=key, expected_len=batch_size)
    return str(values[index])


def _finite_float(value: torch.Tensor) -> float:
    number = float(value.detach().cpu().item())
    if not math.isfinite(number):
        raise ValueError(f"signal logging produced a non-finite value: {number!r}")
    return number


@dataclass
class SampleSignalLogger:
    """Append-only in-memory signal logger.

    The logger records observations as plain dataclass rows and serializes them
    to a JSON-safe payload for experiment artifacts. It intentionally does not
    mutate rows after writing; forgetting labels should be derived later.
    """

    rows: list[SampleSignalRow] = field(default_factory=list)

    def log_batch(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch: dict[str, Any],
        observation_type: str,
        trained_task_id: int,
        global_step: int,
        epoch: int | None = None,
        evaluated_task_id: int | None = None,
        is_replay: bool | None = None,
    ) -> None:
        """Record per-sample loss, confidence, and replay metadata for a batch."""

        if logits.ndim != 2:
            raise ValueError("signal logging expects logits with shape [batch, classes]")
        if targets.ndim != 1:
            targets = targets.reshape(-1)
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets have different batch sizes")

        batch_size = int(targets.shape[0])
        if batch_size == 0:
            return

        detached_logits = logits.detach()
        detached_targets = targets.detach().long().to(detached_logits.device)
        losses = F.cross_entropy(detached_logits, detached_targets, reduction="none")
        probabilities = F.softmax(detached_logits, dim=1)
        confidence_values, predictions = probabilities.max(dim=1)
        target_probabilities = probabilities.gather(1, detached_targets.view(-1, 1)).squeeze(1)
        entropy_values = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)

        for index in range(batch_size):
            confidence = _finite_float(confidence_values[index])
            row_is_replay = (
                bool(is_replay)
                if is_replay is not None
                else _bool_field(batch, "is_replay", index, batch_size, default=False)
            )
            replay_count = _optional_int_field(
                batch,
                "replay_count",
                index,
                batch_size,
                default=0,
            )
            last_replay_step = _optional_int_field(
                batch,
                "last_replay_step",
                index,
                batch_size,
                default=None,
            )
            predicted_class = int(predictions[index].detach().cpu().item())
            target = int(detached_targets[index].detach().cpu().item())
            self.rows.append(
                SampleSignalRow(
                    sample_id=_int_field(batch, "sample_id", index, batch_size),
                    source_task_id=_int_field(batch, "task_id", index, batch_size),
                    original_class_id=_int_field(
                        batch,
                        "original_class_id",
                        index,
                        batch_size,
                    ),
                    within_task_label=_int_field(
                        batch,
                        "within_task_label",
                        index,
                        batch_size,
                    ),
                    original_index=_int_field(batch, "original_index", index, batch_size),
                    split=_str_field(batch, "split", index, batch_size),
                    target=target,
                    observation_type=observation_type,
                    trained_task_id=int(trained_task_id),
                    evaluated_task_id=(
                        int(evaluated_task_id) if evaluated_task_id is not None else None
                    ),
                    epoch=int(epoch) if epoch is not None else None,
                    global_step=int(global_step),
                    is_replay=row_is_replay,
                    replay_count=int(replay_count or 0),
                    last_replay_step=last_replay_step,
                    loss=_finite_float(losses[index]),
                    predicted_class=predicted_class,
                    correct=predicted_class == target,
                    confidence=confidence,
                    target_probability=_finite_float(target_probabilities[index]),
                    uncertainty=1.0 - confidence,
                    entropy=_finite_float(entropy_values[index]),
                )
            )

    def summary(self) -> dict[str, Any]:
        """Return compact counts suitable for run metrics."""

        counts_by_observation_type: dict[str, int] = {}
        replay_rows = 0
        sample_ids: set[int] = set()
        for row in self.rows:
            counts_by_observation_type[row.observation_type] = (
                counts_by_observation_type.get(row.observation_type, 0) + 1
            )
            replay_rows += int(row.is_replay)
            sample_ids.add(row.sample_id)

        return {
            "schema_version": SIGNAL_SCHEMA_VERSION,
            "row_count": len(self.rows),
            "unique_sample_count": len(sample_ids),
            "replay_observation_count": replay_rows,
            "counts_by_observation_type": counts_by_observation_type,
            "fields": list(SIGNAL_FIELDS),
        }

    def to_json_payload(self) -> dict[str, Any]:
        """Return a JSON-safe artifact payload."""

        return {
            "schema_version": SIGNAL_SCHEMA_VERSION,
            "fields": list(SIGNAL_FIELDS),
            "summary": self.summary(),
            "rows": [asdict(row) for row in self.rows],
        }
