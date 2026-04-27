"""Training loops for continual-learning experiments."""

from src.training.continual import (
    ContinualTrainerConfig,
    ContinualTrainingResult,
    evaluate_task_accuracy,
    train_continual,
)

__all__ = [
    "ContinualTrainerConfig",
    "ContinualTrainingResult",
    "evaluate_task_accuracy",
    "train_continual",
]
