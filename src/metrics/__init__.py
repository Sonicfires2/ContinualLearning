"""Metrics for continual-learning experiments."""

from src.metrics.continual import (
    AccuracyMatrix,
    average_accuracy_after_task,
    average_forgetting,
    final_accuracy,
    forgetting_by_task,
)

__all__ = [
    "AccuracyMatrix",
    "average_accuracy_after_task",
    "average_forgetting",
    "final_accuracy",
    "forgetting_by_task",
]
