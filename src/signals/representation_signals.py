"""Representation-drift sample signals for forgetting diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import torch
from torch import nn


REPRESENTATION_SIGNAL_SCHEMA_VERSION = 1
REPRESENTATION_SIGNAL_FIELDS = (
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
    "predicted_class",
    "correct",
    "representation_l2",
    "reference_representation_l2",
    "reference_trained_task_id",
    "reference_global_step",
    "cosine_similarity_to_reference",
    "representation_drift",
)


@dataclass(frozen=True)
class RepresentationSignalRow:
    """One hidden-representation drift observation for one sample."""

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
    predicted_class: int
    correct: bool
    representation_l2: float
    reference_representation_l2: float
    reference_trained_task_id: int
    reference_global_step: int
    cosine_similarity_to_reference: float
    representation_drift: float


@dataclass
class _ReferenceRepresentation:
    vector: torch.Tensor
    trained_task_id: int
    global_step: int
    l2: float


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
            f"representation signal field {field_name!r} has length {len(values)}, "
            f"expected {expected_len}"
        )
    return values


def _int_field(batch: dict[str, Any], key: str, index: int, batch_size: int) -> int:
    if key not in batch:
        raise KeyError(f"representation signal logging requires batch metadata key {key!r}")
    return int(_as_sequence(batch[key], field_name=key, expected_len=batch_size)[index])


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
    return bool(_as_sequence(batch[key], field_name=key, expected_len=batch_size)[index])


def _str_field(batch: dict[str, Any], key: str, index: int, batch_size: int) -> str:
    if key not in batch:
        raise KeyError(f"representation signal logging requires batch metadata key {key!r}")
    return str(_as_sequence(batch[key], field_name=key, expected_len=batch_size)[index])


def _finite_float(value: torch.Tensor | float) -> float:
    number = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise ValueError(f"representation signal logging produced a non-finite value: {number!r}")
    return number


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim <= 1:
        return features.reshape(features.shape[0], -1)
    return torch.flatten(features, start_dim=1)


def _sequential_modules_before_last_linear(model: nn.Module) -> tuple[list[nn.Module], nn.Linear]:
    network = getattr(model, "network", None)
    if not isinstance(network, nn.Sequential):
        raise TypeError(
            "representation drift signals currently require a model with a "
            "Sequential `network` attribute"
        )
    modules = list(network)
    linear_indices = [index for index, module in enumerate(modules) if isinstance(module, nn.Linear)]
    if not linear_indices:
        raise TypeError("model network does not contain a Linear classifier layer")
    last_index = linear_indices[-1]
    return modules[:last_index], modules[last_index]


def representation_signal_tensors(
    *,
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return logits and penultimate representations for a batch."""

    prefix_modules, classifier = _sequential_modules_before_last_linear(model)
    with torch.no_grad():
        features = inputs
        for module in prefix_modules:
            features = module(features)
        representations = _flatten_features(features)
        logits = classifier(representations)
        representation_l2 = torch.linalg.vector_norm(representations, ord=2, dim=1)
    return {
        "logits": logits,
        "representations": representations,
        "representation_l2": representation_l2,
    }


def _cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    first_norm = torch.linalg.vector_norm(first, ord=2)
    second_norm = torch.linalg.vector_norm(second, ord=2)
    if float(first_norm.item()) == 0.0 and float(second_norm.item()) == 0.0:
        return 1.0
    if float(first_norm.item()) == 0.0 or float(second_norm.item()) == 0.0:
        return 0.0
    similarity = torch.dot(first, second) / (first_norm * second_norm)
    return max(-1.0, min(1.0, _finite_float(similarity)))


@dataclass
class RepresentationDriftLogger:
    """Append-only logger for cosine drift from each sample's first representation."""

    rows: list[RepresentationSignalRow] = field(default_factory=list)
    references: dict[int, _ReferenceRepresentation] = field(default_factory=dict)

    def log_batch(
        self,
        *,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        batch: dict[str, Any],
        observation_type: str,
        trained_task_id: int,
        global_step: int,
        epoch: int | None = None,
        evaluated_task_id: int | None = None,
        is_replay: bool | None = None,
    ) -> None:
        """Record representation drift for a batch.

        The first observed representation for each sample is treated as the
        reference. For the project diagnostics this occurs at the sample's first
        seen-task evaluation, which is the evaluation-compatible proxy for
        insertion-time representation.
        """

        if targets.ndim != 1:
            targets = targets.reshape(-1)
        batch_size = int(targets.shape[0])
        if batch_size == 0:
            return
        tensors = representation_signal_tensors(model=model, inputs=inputs)
        predictions = tensors["logits"].argmax(dim=1)
        representations = tensors["representations"].detach().cpu()
        representation_l2 = tensors["representation_l2"].detach().cpu()

        for index in range(batch_size):
            sample_id = _int_field(batch, "sample_id", index, batch_size)
            current_vector = representations[index].clone()
            current_l2 = _finite_float(representation_l2[index])
            if sample_id not in self.references:
                self.references[sample_id] = _ReferenceRepresentation(
                    vector=current_vector,
                    trained_task_id=int(trained_task_id),
                    global_step=int(global_step),
                    l2=current_l2,
                )
            reference = self.references[sample_id]
            similarity = _cosine_similarity(current_vector, reference.vector)
            drift = 1.0 - similarity
            target = int(targets[index].detach().cpu().item())
            predicted_class = int(predictions[index].detach().cpu().item())
            row_is_replay = (
                bool(is_replay)
                if is_replay is not None
                else _bool_field(batch, "is_replay", index, batch_size, default=False)
            )
            self.rows.append(
                RepresentationSignalRow(
                    sample_id=sample_id,
                    source_task_id=_int_field(batch, "task_id", index, batch_size),
                    original_class_id=_int_field(batch, "original_class_id", index, batch_size),
                    within_task_label=_int_field(batch, "within_task_label", index, batch_size),
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
                    predicted_class=predicted_class,
                    correct=predicted_class == target,
                    representation_l2=current_l2,
                    reference_representation_l2=reference.l2,
                    reference_trained_task_id=reference.trained_task_id,
                    reference_global_step=reference.global_step,
                    cosine_similarity_to_reference=similarity,
                    representation_drift=drift,
                )
            )

    def summary(self) -> dict[str, Any]:
        counts_by_observation_type: dict[str, int] = {}
        sample_ids: set[int] = set()
        drift_values = []
        for row in self.rows:
            counts_by_observation_type[row.observation_type] = (
                counts_by_observation_type.get(row.observation_type, 0) + 1
            )
            sample_ids.add(row.sample_id)
            drift_values.append(row.representation_drift)
        return {
            "schema_version": REPRESENTATION_SIGNAL_SCHEMA_VERSION,
            "row_count": len(self.rows),
            "unique_sample_count": len(sample_ids),
            "reference_count": len(self.references),
            "counts_by_observation_type": counts_by_observation_type,
            "fields": list(REPRESENTATION_SIGNAL_FIELDS),
            "signal_definition": "cosine distance between current penultimate representation and the sample's first observed representation",
            "mean_representation_drift": (
                sum(drift_values) / len(drift_values) if drift_values else None
            ),
            "max_representation_drift": max(drift_values) if drift_values else None,
        }

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REPRESENTATION_SIGNAL_SCHEMA_VERSION,
            "fields": list(REPRESENTATION_SIGNAL_FIELDS),
            "definition": {
                "signal_family": "representation_drift",
                "representation_scope": "penultimate_pre_classifier_activation",
                "formula": "1 - cosine_similarity(h_i(reference), h_i(current))",
                "reference_rule": "first observed representation for each sample",
                "future_leakage_guard": "rows are logged at the same observation time as ordinary sample signals",
            },
            "summary": self.summary(),
            "rows": [asdict(row) for row in self.rows],
        }
