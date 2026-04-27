"""Metric helpers for accuracy matrices from continual-learning runs."""

from __future__ import annotations

from typing import TypeAlias


AccuracyMatrix: TypeAlias = list[list[float | None]]


def _seen_values(row: list[float | None]) -> list[float]:
    return [float(value) for value in row if value is not None]


def average_accuracy_after_task(
    accuracy_matrix: AccuracyMatrix,
    task_index: int,
) -> float:
    """Mean accuracy over tasks seen after completing `task_index`."""

    if task_index < 0 or task_index >= len(accuracy_matrix):
        raise IndexError("task_index is outside the accuracy matrix")
    values = _seen_values(accuracy_matrix[task_index])
    if not values:
        raise ValueError("task row does not contain any seen-task accuracies")
    return sum(values) / len(values)


def final_accuracy(accuracy_matrix: AccuracyMatrix) -> float:
    """Mean accuracy over all tasks after the final training task."""

    if not accuracy_matrix:
        raise ValueError("accuracy_matrix must not be empty")
    return average_accuracy_after_task(accuracy_matrix, len(accuracy_matrix) - 1)


def forgetting_by_task(accuracy_matrix: AccuracyMatrix) -> dict[int, float]:
    """Return per-task forgetting using best historical accuracy minus final accuracy.

    The final task is excluded because no later task has had a chance to
    interfere with it.
    """

    if not accuracy_matrix:
        raise ValueError("accuracy_matrix must not be empty")

    final_row = accuracy_matrix[-1]
    forgetting: dict[int, float] = {}
    for task_id in range(max(0, len(accuracy_matrix) - 1)):
        historical_values = [
            row[task_id]
            for row in accuracy_matrix[task_id:]
            if row[task_id] is not None
        ]
        final_value = final_row[task_id]
        if not historical_values or final_value is None:
            raise ValueError(f"accuracy_matrix is missing values for task {task_id}")
        forgetting[task_id] = max(float(value) for value in historical_values) - float(
            final_value
        )
    return forgetting


def average_forgetting(accuracy_matrix: AccuracyMatrix) -> float:
    """Mean forgetting over all non-final tasks."""

    forgetting = forgetting_by_task(accuracy_matrix)
    if not forgetting:
        return 0.0
    return sum(forgetting.values()) / len(forgetting)
