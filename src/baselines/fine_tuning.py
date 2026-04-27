"""Naive sequential fine-tuning baseline.

This baseline intentionally contains no replay and no forgetting-aware logic.
It establishes the no-intervention reference point needed before random replay
or spaced replay can make a credible claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

from src.data.split_cifar100 import (
    SplitCIFAR100TaskStream,
    SplitCIFAR100TaskStreamConfig,
    load_cifar100_dataset,
)
from src.experiment_tracking import ArtifactPaths, ExperimentRunConfig, save_experiment_artifacts
from src.models import build_mlp, count_parameters
from src.signals import SIGNAL_FIELDS, SampleSignalLogger
from src.training import ContinualTrainerConfig, ContinualTrainingResult, train_continual


class BaselineDataUnavailableError(RuntimeError):
    """Raised when a real baseline run cannot find its required dataset."""


@dataclass(frozen=True)
class FineTuningBaselineConfig:
    """Configuration for the no-replay sequential fine-tuning baseline."""

    protocol_id: str = "core_split_cifar100_v2"
    method_name: str = "fine_tuning"
    run_name: str = "fine_tuning_baseline"
    seed: int = 0
    output_root: str = "experiments/runs"
    overwrite: bool = False
    smoke: bool = False
    data_root: str = "data"
    download: bool = False
    task_count: int = 10
    classes_per_task: int = 10
    split_seed: int = 0
    epochs_per_task: int = 1
    batch_size: int = 32
    eval_batch_size: int = 128
    learning_rate: float = 0.01
    device: str = "auto"
    hidden_dims: tuple[int, ...] = (256,)
    dropout: float = 0.0
    target_key: str = "original_class_id"
    log_signals: bool = True


@dataclass(frozen=True)
class FineTuningBaselineRun:
    """In-memory result plus saved artifact locations."""

    result: ContinualTrainingResult
    artifacts: ArtifactPaths
    run_config: ExperimentRunConfig


class FixtureVisionDataset(Dataset):
    """Small deterministic image-like fixture for offline smoke tests."""

    def __init__(
        self,
        *,
        task_count: int = 3,
        classes_per_task: int = 2,
        examples_per_class: int = 4,
    ) -> None:
        self.task_count = task_count
        self.classes_per_task = classes_per_task
        self.examples_per_class = examples_per_class
        self.class_count = task_count * classes_per_task
        self.targets: list[int] = []
        self._features: list[torch.Tensor] = []

        for class_id in range(self.class_count):
            for example_id in range(examples_per_class):
                self.targets.append(class_id)
                self._features.append(self._feature_for(class_id, example_id))

    def _feature_for(self, class_id: int, example_id: int) -> torch.Tensor:
        image = torch.zeros(1, 4, 4, dtype=torch.float32)
        row = class_id // 4
        column = class_id % 4
        image[0, row, column] = 1.0
        image[0, (row + 1) % 4, column] = 0.1 * example_id
        return image

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._features[index], self.targets[index]


def _normalization_transform():
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("torchvision is required for real CIFAR-100 runs") from exc

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )


def build_fixture_streams(
    config: FineTuningBaselineConfig,
) -> tuple[SplitCIFAR100TaskStream, SplitCIFAR100TaskStream, tuple[int, ...], tuple[int, ...]]:
    """Build offline train/eval streams for pipeline verification."""

    dataset = FixtureVisionDataset(
        task_count=config.task_count,
        classes_per_task=config.classes_per_task,
    )
    class_order = tuple(range(config.task_count * config.classes_per_task))
    train_stream = SplitCIFAR100TaskStream.from_dataset(
        dataset=dataset,
        task_count=config.task_count,
        classes_per_task=config.classes_per_task,
        split_seed=config.split_seed,
        split="train",
        sample_id_offset=0,
        class_order=class_order,
    )
    eval_stream = SplitCIFAR100TaskStream.from_dataset(
        dataset=dataset,
        task_count=config.task_count,
        classes_per_task=config.classes_per_task,
        split_seed=config.split_seed,
        split="test",
        sample_id_offset=1_000_000,
        class_order=class_order,
    )
    return train_stream, eval_stream, (1, 4, 4), class_order


def build_real_split_cifar100_streams(
    config: FineTuningBaselineConfig,
) -> tuple[SplitCIFAR100TaskStream, SplitCIFAR100TaskStream, tuple[int, ...], tuple[int, ...]]:
    transform = _normalization_transform()
    dataset_config = SplitCIFAR100TaskStreamConfig(
        data_root=config.data_root,
        download=config.download,
        task_count=config.task_count,
        classes_per_task=config.classes_per_task,
        split_seed=config.split_seed,
    )
    try:
        train_dataset = load_cifar100_dataset(
            data_root=dataset_config.data_root,
            train=True,
            download=dataset_config.download,
            transform=transform,
        )
    except RuntimeError as exc:
        raise BaselineDataUnavailableError(
            "Split CIFAR-100 training data is not available. "
            f"data_root={config.data_root!r}, download={config.download!r}. "
            "Populate the dataset first or set download: true for an approved data run."
        ) from exc
    class_order = tuple(
        class_id
        for task_spec in SplitCIFAR100TaskStream.from_dataset(
            dataset=train_dataset,
            task_count=config.task_count,
            classes_per_task=config.classes_per_task,
            split_seed=config.split_seed,
            split="train",
            sample_id_offset=dataset_config.train_sample_id_offset,
        ).task_specs
        for class_id in task_spec.class_ids
    )
    train_stream = SplitCIFAR100TaskStream.from_dataset(
        dataset=train_dataset,
        task_count=config.task_count,
        classes_per_task=config.classes_per_task,
        split_seed=config.split_seed,
        split="train",
        sample_id_offset=dataset_config.train_sample_id_offset,
        class_order=class_order,
    )
    try:
        eval_dataset = load_cifar100_dataset(
            data_root=dataset_config.data_root,
            train=False,
            download=dataset_config.download,
            transform=transform,
        )
    except RuntimeError as exc:
        raise BaselineDataUnavailableError(
            "Split CIFAR-100 evaluation data is not available. "
            f"data_root={config.data_root!r}, download={config.download!r}. "
            "Populate the dataset first or set download: true for an approved data run."
        ) from exc
    eval_stream = SplitCIFAR100TaskStream.from_dataset(
        dataset=eval_dataset,
        task_count=config.task_count,
        classes_per_task=config.classes_per_task,
        split_seed=config.split_seed,
        split="test",
        sample_id_offset=dataset_config.test_sample_id_offset,
        class_order=class_order,
    )
    return train_stream, eval_stream, (3, 32, 32), class_order


def _experiment_run_config(
    *,
    config: FineTuningBaselineConfig,
    model_parameter_count: int,
    class_order: tuple[int, ...],
) -> ExperimentRunConfig:
    return ExperimentRunConfig(
        protocol_id=config.protocol_id,
        method_name=config.method_name,
        seed=config.seed,
        run_name=config.run_name,
        dataset={
            "name": "fixture_split_cifar100" if config.smoke else "split_cifar100",
            "data_root": config.data_root,
            "download": config.download,
            "task_count": config.task_count,
            "classes_per_task": config.classes_per_task,
            "split_seed": config.split_seed,
            "target_key": config.target_key,
            "smoke": config.smoke,
        },
        model={
            "name": "flatten_mlp",
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "output_dim": config.task_count * config.classes_per_task,
            "trainable_parameters": model_parameter_count,
        },
        trainer={
            "epochs_per_task": config.epochs_per_task,
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "learning_rate": config.learning_rate,
            "device": config.device,
            "target_key": config.target_key,
        },
        evaluation={
            "schedule": "evaluate_all_seen_tasks_after_each_task",
            "metrics": ["final_accuracy", "average_forgetting", "average_accuracy"],
        },
        task_split={
            "class_order": list(class_order),
        },
        method={
            "description": "naive sequential fine-tuning without replay",
            "uses_replay": False,
        },
        replay={
            "enabled": False,
            "buffer_capacity": 0,
        },
        signals={
            "enabled": config.log_signals,
            "artifact": "sample_signals.json" if config.log_signals else None,
            "fields": list(SIGNAL_FIELDS) if config.log_signals else [],
            "observation_types": (
                ["current_train", "seen_task_eval"] if config.log_signals else []
            ),
        },
    )


def run_fine_tuning_baseline(config: FineTuningBaselineConfig) -> FineTuningBaselineRun:
    """Run the no-replay baseline and save complete experiment artifacts."""

    if config.target_key != "original_class_id":
        raise ValueError(
            "fine-tuning baseline defaults to class-incremental original_class_id targets; "
            "change this only with an explicit research justification"
        )

    if config.smoke:
        train_stream, eval_stream, input_shape, class_order = build_fixture_streams(config)
    else:
        train_stream, eval_stream, input_shape, class_order = build_real_split_cifar100_streams(config)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = build_mlp(
        input_shape=input_shape,
        output_dim=config.task_count * config.classes_per_task,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    trainer_config = ContinualTrainerConfig(
        epochs_per_task=config.epochs_per_task,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        device=config.device,
        seed=config.seed,
        target_key=config.target_key,
    )
    signal_logger = SampleSignalLogger() if config.log_signals else None
    result = train_continual(
        model=model,
        train_stream=train_stream,
        eval_stream=eval_stream,
        optimizer=optimizer,
        config=trainer_config,
        signal_logger=signal_logger,
    )
    if signal_logger is not None:
        result.method_metrics["signals"] = signal_logger.summary()
    run_config = _experiment_run_config(
        config=config,
        model_parameter_count=count_parameters(model),
        class_order=class_order,
    )
    artifacts = save_experiment_artifacts(
        output_root=config.output_root,
        run_config=run_config,
        result=result,
        overwrite=config.overwrite,
        extra_metadata={
            "baseline_status": "smoke" if config.smoke else "real_split_cifar100",
            "research_role": "no_replay_reference",
        },
        extra_json_artifacts=(
            {"sample_signals": signal_logger.to_json_payload()}
            if signal_logger is not None
            else None
        ),
    )
    return FineTuningBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> FineTuningBaselineConfig:
    """Load a fine-tuning baseline config from YAML."""

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "fine_tuning_baseline" not in raw:
        raise ValueError("config must contain a fine_tuning_baseline section")
    section = raw["fine_tuning_baseline"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return FineTuningBaselineConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the no-replay fine-tuning baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        run = run_fine_tuning_baseline(config)
    except BaselineDataUnavailableError as exc:
        print(f"Baseline data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved fine-tuning baseline artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
