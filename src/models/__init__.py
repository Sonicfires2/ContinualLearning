"""Model factories for continual-learning experiments."""

from src.models.simple import FlattenMLP, MLPConfig, build_mlp, count_parameters

__all__ = [
    "FlattenMLP",
    "MLPConfig",
    "build_mlp",
    "count_parameters",
]
