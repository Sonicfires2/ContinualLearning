"""Small baseline models used before stronger architectures are introduced."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Sequence

from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for a flattening multilayer perceptron."""

    input_shape: tuple[int, ...]
    output_dim: int
    hidden_dims: tuple[int, ...] = (256,)
    dropout: float = 0.0


class FlattenMLP(nn.Module):
    """Flatten arbitrary input tensors and classify with an MLP."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        if config.output_dim < 1:
            raise ValueError("output_dim must be positive")
        if any(dim < 1 for dim in config.input_shape):
            raise ValueError("input_shape dimensions must be positive")
        if any(dim < 1 for dim in config.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        input_dim = reduce(mul, config.input_shape, 1)
        layers: list[nn.Module] = [nn.Flatten()]
        previous_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, config.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_mlp(
    *,
    input_shape: Sequence[int],
    output_dim: int,
    hidden_dims: Sequence[int] = (256,),
    dropout: float = 0.0,
) -> FlattenMLP:
    """Build the default lightweight baseline classifier."""

    return FlattenMLP(
        MLPConfig(
            input_shape=tuple(int(dim) for dim in input_shape),
            output_dim=int(output_dim),
            hidden_dims=tuple(int(dim) for dim in hidden_dims),
            dropout=float(dropout),
        )
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for experiment metadata."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
