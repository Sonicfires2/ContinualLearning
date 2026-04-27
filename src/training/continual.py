"""Method-agnostic continual training and seen-task evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from time import perf_counter
from typing import Any, Callable, Protocol, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.metrics.continual import AccuracyMatrix
from src.signals import GradientSignalLogger, SampleSignalLogger


class TaskStreamLike(Protocol):
    """Minimal protocol consumed by the continual trainer."""

    def __len__(self) -> int:
        """Return the number of tasks."""

    def task_dataset(self, task_id: int) -> Dataset:
        """Return a dataset for one task."""


@dataclass(frozen=True)
class ContinualTrainerConfig:
    """Configuration for a method-agnostic continual training loop."""

    epochs_per_task: int = 1
    batch_size: int = 32
    eval_batch_size: int = 128
    device: str = "cpu"
    seed: int = 0
    shuffle_train: bool = True
    target_key: str = "y"
    num_workers: int = 0


@dataclass(frozen=True)
class ContinualTrainingResult:
    """Outputs required by later metrics, baselines, and reports."""

    accuracy_matrix: AccuracyMatrix
    train_losses: list[float] = field(default_factory=list)
    training_time_seconds: float = 0.0
    task_count: int = 0
    method_metrics: dict[str, Any] = field(default_factory=dict)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]
    if isinstance(model_output, (tuple, list)) and model_output:
        first = model_output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError("model output must be a tensor, a logits dict, or a tuple/list")


def _batch_to_tensors(
    batch: dict[str, Any],
    *,
    target_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "x" not in batch:
        raise KeyError("batch is missing required key 'x'")
    if target_key not in batch:
        raise KeyError(f"batch is missing target key {target_key!r}")

    x = batch["x"]
    y = batch[target_key]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    return x.to(device), y.long().to(device)


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def evaluate_task_accuracy(
    *,
    model: nn.Module,
    dataset: Dataset,
    config: ContinualTrainerConfig,
    device: torch.device | None = None,
    signal_logger: SampleSignalLogger | None = None,
    gradient_signal_logger: GradientSignalLogger | None = None,
    trained_task_id: int | None = None,
    evaluated_task_id: int | None = None,
    global_step: int = 0,
) -> float:
    """Evaluate one task dataset and return classification accuracy."""

    resolved_device = device or _resolve_device(config.device)
    model.eval()
    loader = _make_loader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _batch_to_tensors(
                batch,
                target_key=config.target_key,
                device=resolved_device,
            )
            logits = _extract_logits(model(x))
            if signal_logger is not None:
                if trained_task_id is None or evaluated_task_id is None:
                    raise ValueError(
                        "trained_task_id and evaluated_task_id are required "
                        "when logging evaluation signals"
                    )
                signal_logger.log_batch(
                    logits=logits,
                    targets=y,
                    batch=batch,
                    observation_type="seen_task_eval",
                    trained_task_id=trained_task_id,
                    evaluated_task_id=evaluated_task_id,
                    global_step=global_step,
                    is_replay=False,
                )
            if gradient_signal_logger is not None:
                if trained_task_id is None or evaluated_task_id is None:
                    raise ValueError(
                        "trained_task_id and evaluated_task_id are required "
                        "when logging gradient evaluation signals"
                    )
                gradient_signal_logger.log_batch(
                    model=model,
                    inputs=x,
                    targets=y,
                    batch=batch,
                    observation_type="seen_task_eval",
                    trained_task_id=trained_task_id,
                    evaluated_task_id=evaluated_task_id,
                    global_step=global_step,
                    is_replay=False,
                )
            predictions = logits.argmax(dim=1)
            correct += int((predictions == y).sum().item())
            total += int(y.numel())

    if total == 0:
        raise ValueError("cannot evaluate accuracy on an empty dataset")
    return correct / total


def train_continual(
    *,
    model: nn.Module,
    train_stream: TaskStreamLike,
    optimizer: torch.optim.Optimizer,
    config: ContinualTrainerConfig = ContinualTrainerConfig(),
    eval_stream: TaskStreamLike | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    signal_logger: SampleSignalLogger | None = None,
) -> ContinualTrainingResult:
    """Train sequentially over tasks and evaluate all seen tasks after each task.

    This loop is intentionally method-agnostic. Replay, signal logging, and
    scheduler-specific behavior should plug in later without changing the
    accuracy-matrix contract produced here.
    """

    if len(train_stream) < 1:
        raise ValueError("train_stream must contain at least one task")
    if config.epochs_per_task < 0:
        raise ValueError("epochs_per_task must be non-negative")
    if config.batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")

    _seed_everything(config.seed)
    resolved_device = _resolve_device(config.device)
    model.to(resolved_device)
    criterion = loss_fn or nn.CrossEntropyLoss()
    evaluation_stream = eval_stream or train_stream

    task_count = len(train_stream)
    accuracy_matrix: AccuracyMatrix = [
        [None for _ in range(task_count)] for _ in range(task_count)
    ]
    train_losses: list[float] = []
    global_step = 0

    start_time = perf_counter()
    for task_id in range(task_count):
        model.train()
        task_dataset = train_stream.task_dataset(task_id)
        loader = _make_loader(
            task_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
            seed=config.seed + task_id,
            num_workers=config.num_workers,
        )

        for epoch in range(config.epochs_per_task):
            for batch in loader:
                x, y = _batch_to_tensors(
                    batch,
                    target_key=config.target_key,
                    device=resolved_device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = _extract_logits(model(x))
                loss = criterion(logits, y)
                if signal_logger is not None:
                    signal_logger.log_batch(
                        logits=logits,
                        targets=y,
                        batch=batch,
                        observation_type="current_train",
                        trained_task_id=task_id,
                        epoch=epoch,
                        global_step=global_step,
                        is_replay=False,
                    )
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
                global_step += 1

        for eval_task_id in range(task_id + 1):
            accuracy_matrix[task_id][eval_task_id] = evaluate_task_accuracy(
                model=model,
                dataset=evaluation_stream.task_dataset(eval_task_id),
                config=config,
                device=resolved_device,
                signal_logger=signal_logger,
                trained_task_id=task_id,
                evaluated_task_id=eval_task_id,
                global_step=global_step,
            )

    return ContinualTrainingResult(
        accuracy_matrix=accuracy_matrix,
        train_losses=train_losses,
        training_time_seconds=perf_counter() - start_time,
        task_count=task_count,
    )
