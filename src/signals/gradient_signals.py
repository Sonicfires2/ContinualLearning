"""Gradient-family sample signals for forgetting diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


GRADIENT_SIGNAL_SCHEMA_VERSION = 1
GRADIENT_SIGNAL_FIELDS = (
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
    "loss",
    "predicted_class",
    "correct",
    "target_probability",
    "logit_gradient_l2",
    "penultimate_activation_l2",
    "classifier_weight_gradient_l2",
    "classifier_bias_gradient_l2",
    "last_layer_gradient_l2",
)


@dataclass(frozen=True)
class GradientSignalRow:
    """One exact final-layer gradient signal for one observed sample."""

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
    loss: float
    predicted_class: int
    correct: bool
    target_probability: float
    logit_gradient_l2: float
    penultimate_activation_l2: float
    classifier_weight_gradient_l2: float
    classifier_bias_gradient_l2: float
    last_layer_gradient_l2: float


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
            f"gradient signal field {field_name!r} has length {len(values)}, "
            f"expected {expected_len}"
        )
    return values


def _int_field(batch: dict[str, Any], key: str, index: int, batch_size: int) -> int:
    if key not in batch:
        raise KeyError(f"gradient signal logging requires batch metadata key {key!r}")
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
        raise KeyError(f"gradient signal logging requires batch metadata key {key!r}")
    return str(_as_sequence(batch[key], field_name=key, expected_len=batch_size)[index])


def _finite_float(value: torch.Tensor | float) -> float:
    number = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise ValueError(f"gradient signal logging produced a non-finite value: {number!r}")
    return number


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim <= 1:
        return features.reshape(features.shape[0], -1)
    return torch.flatten(features, start_dim=1)


def _sequential_modules_before_last_linear(model: nn.Module) -> tuple[list[nn.Module], nn.Linear]:
    network = getattr(model, "network", None)
    if not isinstance(network, nn.Sequential):
        raise TypeError(
            "last-layer gradient signals currently require a model with a "
            "Sequential `network` attribute"
        )
    modules = list(network)
    linear_indices = [index for index, module in enumerate(modules) if isinstance(module, nn.Linear)]
    if not linear_indices:
        raise TypeError("model network does not contain a Linear classifier layer")
    last_index = linear_indices[-1]
    return modules[:last_index], modules[last_index]


def last_layer_gradient_signal_tensors(
    *,
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute exact per-sample final-layer cross-entropy gradient norms.

    This avoids a per-sample backward pass by using the closed form of the last
    linear layer gradient: outer(probabilities - one_hot(target), features).
    """

    if targets.ndim != 1:
        targets = targets.reshape(-1)
    prefix_modules, classifier = _sequential_modules_before_last_linear(model)
    with torch.no_grad():
        features = inputs
        for module in prefix_modules:
            features = module(features)
        flat_features = _flatten_features(features)
        logits = classifier(flat_features)
        target_values = targets.long().to(logits.device)
        losses = F.cross_entropy(logits, target_values, reduction="none")
        probabilities = F.softmax(logits, dim=1)
        target_probabilities = probabilities.gather(
            1,
            target_values.view(-1, 1),
        ).squeeze(1)
        predictions = logits.argmax(dim=1)
        logit_gradient = probabilities.clone()
        logit_gradient[
            torch.arange(logit_gradient.shape[0], device=logit_gradient.device),
            target_values,
        ] -= 1.0
        logit_gradient_l2 = torch.linalg.vector_norm(logit_gradient, ord=2, dim=1)
        activation_l2 = torch.linalg.vector_norm(flat_features, ord=2, dim=1)
        weight_gradient_l2 = logit_gradient_l2 * activation_l2
        bias_gradient_l2 = logit_gradient_l2
        last_layer_gradient_l2 = torch.sqrt(
            weight_gradient_l2.square() + bias_gradient_l2.square()
        )
    return {
        "logits": logits,
        "loss": losses,
        "predicted_class": predictions,
        "correct": predictions == target_values,
        "target_probability": target_probabilities,
        "logit_gradient_l2": logit_gradient_l2,
        "penultimate_activation_l2": activation_l2,
        "classifier_weight_gradient_l2": weight_gradient_l2,
        "classifier_bias_gradient_l2": bias_gradient_l2,
        "last_layer_gradient_l2": last_layer_gradient_l2,
    }


@dataclass
class GradientSignalLogger:
    """Append-only logger for expensive gradient-family signals."""

    rows: list[GradientSignalRow] = field(default_factory=list)

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
        """Record last-layer gradient signals for a batch."""

        if targets.ndim != 1:
            targets = targets.reshape(-1)
        batch_size = int(targets.shape[0])
        if batch_size == 0:
            return
        tensors = last_layer_gradient_signal_tensors(
            model=model,
            inputs=inputs,
            targets=targets,
        )
        for index in range(batch_size):
            target = int(targets[index].detach().cpu().item())
            predicted_class = int(tensors["predicted_class"][index].detach().cpu().item())
            row_is_replay = (
                bool(is_replay)
                if is_replay is not None
                else _bool_field(batch, "is_replay", index, batch_size, default=False)
            )
            self.rows.append(
                GradientSignalRow(
                    sample_id=_int_field(batch, "sample_id", index, batch_size),
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
                    loss=_finite_float(tensors["loss"][index]),
                    predicted_class=predicted_class,
                    correct=bool(tensors["correct"][index].detach().cpu().item()),
                    target_probability=_finite_float(tensors["target_probability"][index]),
                    logit_gradient_l2=_finite_float(tensors["logit_gradient_l2"][index]),
                    penultimate_activation_l2=_finite_float(
                        tensors["penultimate_activation_l2"][index]
                    ),
                    classifier_weight_gradient_l2=_finite_float(
                        tensors["classifier_weight_gradient_l2"][index]
                    ),
                    classifier_bias_gradient_l2=_finite_float(
                        tensors["classifier_bias_gradient_l2"][index]
                    ),
                    last_layer_gradient_l2=_finite_float(
                        tensors["last_layer_gradient_l2"][index]
                    ),
                )
            )

    def summary(self) -> dict[str, Any]:
        counts_by_observation_type: dict[str, int] = {}
        sample_ids: set[int] = set()
        gradient_values = []
        for row in self.rows:
            counts_by_observation_type[row.observation_type] = (
                counts_by_observation_type.get(row.observation_type, 0) + 1
            )
            sample_ids.add(row.sample_id)
            gradient_values.append(row.last_layer_gradient_l2)
        return {
            "schema_version": GRADIENT_SIGNAL_SCHEMA_VERSION,
            "row_count": len(self.rows),
            "unique_sample_count": len(sample_ids),
            "counts_by_observation_type": counts_by_observation_type,
            "fields": list(GRADIENT_SIGNAL_FIELDS),
            "signal_definition": "exact cross-entropy gradient L2 norm for the final linear layer",
            "mean_last_layer_gradient_l2": (
                sum(gradient_values) / len(gradient_values) if gradient_values else None
            ),
            "max_last_layer_gradient_l2": max(gradient_values) if gradient_values else None,
        }

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GRADIENT_SIGNAL_SCHEMA_VERSION,
            "fields": list(GRADIENT_SIGNAL_FIELDS),
            "definition": {
                "signal_family": "gradient_norm",
                "gradient_scope": "final_linear_layer_parameters",
                "formula": "||grad_W CE|| and ||grad_b CE|| from probabilities - one_hot(target)",
                "future_leakage_guard": "rows are logged at the same observation time as ordinary sample signals",
            },
            "summary": self.summary(),
            "rows": [asdict(row) for row in self.rows],
        }
