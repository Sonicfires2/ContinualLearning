"""Deterministic Split CIFAR-100 task streams with stable sample IDs.

The research plan depends on tracing examples across training, replay, signal
logging, and later forgetting analysis. This module makes that identity explicit
while keeping the task construction testable without downloading CIFAR-100.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterator, Protocol, Sequence


DEFAULT_SAMPLE_ID_OFFSETS = {
    "train": 0,
    "test": 1_000_000,
}


class ClassLabeledDataset(Protocol):
    """Minimal dataset contract for class-incremental task construction."""

    targets: Sequence[int]

    def __len__(self) -> int:
        """Return the number of examples."""

    def __getitem__(self, index: int) -> Any:
        """Return one dataset item."""


@dataclass(frozen=True)
class SplitCIFAR100TaskStreamConfig:
    """Configuration for constructing the locked Split CIFAR-100 stream."""

    data_root: str = "data"
    download: bool = False
    task_count: int = 10
    classes_per_task: int = 10
    split_seed: int = 0
    train_sample_id_offset: int = DEFAULT_SAMPLE_ID_OFFSETS["train"]
    test_sample_id_offset: int = DEFAULT_SAMPLE_ID_OFFSETS["test"]


@dataclass(frozen=True)
class TaskSpec:
    """Description of one sequential class-incremental task."""

    task_id: int
    class_ids: tuple[int, ...]

    def within_task_label(self, original_class_id: int) -> int:
        """Map an original class ID to its task-local label."""

        try:
            return self.class_ids.index(original_class_id)
        except ValueError as exc:
            raise ValueError(
                f"class {original_class_id} is not part of task {self.task_id}"
            ) from exc


@dataclass(frozen=True)
class SampleMetadata:
    """Stable sample metadata attached to every task-stream item."""

    sample_id: int
    task_id: int
    original_class_id: int
    within_task_label: int
    original_index: int
    split: str


def _validate_task_shape(task_count: int, classes_per_task: int) -> None:
    if task_count < 1:
        raise ValueError("task_count must be greater than 0")
    if classes_per_task < 1:
        raise ValueError("classes_per_task must be greater than 0")


def _as_int_targets(targets: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(target) for target in targets)


def _class_order_from_targets(targets: Sequence[int], split_seed: int) -> tuple[int, ...]:
    class_ids = sorted(set(_as_int_targets(targets)))
    rng = Random(split_seed)
    rng.shuffle(class_ids)
    return tuple(class_ids)


def build_task_specs(
    *,
    task_count: int,
    classes_per_task: int,
    split_seed: int = 0,
    targets: Sequence[int] | None = None,
    class_order: Sequence[int] | None = None,
) -> tuple[TaskSpec, ...]:
    """Build deterministic task specifications.

    Use `class_order` when an exact audited order is required. Otherwise, class
    IDs are inferred from `targets` and shuffled deterministically by
    `split_seed`. If neither is supplied, the class universe is assumed to be
    `range(task_count * classes_per_task)`.
    """

    _validate_task_shape(task_count, classes_per_task)
    expected_class_count = task_count * classes_per_task

    if class_order is not None:
        ordered_classes = tuple(int(class_id) for class_id in class_order)
    elif targets is not None:
        ordered_classes = _class_order_from_targets(targets, split_seed)
    else:
        ordered_list = list(range(expected_class_count))
        Random(split_seed).shuffle(ordered_list)
        ordered_classes = tuple(ordered_list)

    if len(ordered_classes) != expected_class_count:
        raise ValueError(
            "task_count * classes_per_task must match the number of classes "
            f"({expected_class_count} expected, {len(ordered_classes)} provided)"
        )
    if len(set(ordered_classes)) != len(ordered_classes):
        raise ValueError("class_order contains duplicate class IDs")

    return tuple(
        TaskSpec(
            task_id=task_id,
            class_ids=ordered_classes[
                task_id * classes_per_task : (task_id + 1) * classes_per_task
            ],
        )
        for task_id in range(task_count)
    )


def _indices_by_class(targets: Sequence[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for index, class_id in enumerate(_as_int_targets(targets)):
        grouped.setdefault(class_id, []).append(index)
    return grouped


def _extract_input(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("x", item.get("image"))
    if isinstance(item, (tuple, list)) and item:
        return item[0]
    return item


class TaskSubsetDataset:
    """Dataset view for a single task with stable sample metadata."""

    def __init__(
        self,
        *,
        dataset: ClassLabeledDataset,
        task_spec: TaskSpec,
        indices: Sequence[int],
        split: str,
        sample_id_offset: int,
    ) -> None:
        self.dataset = dataset
        self.task_spec = task_spec
        self.indices = tuple(int(index) for index in indices)
        self.split = split
        self.sample_id_offset = int(sample_id_offset)
        self._targets = _as_int_targets(dataset.targets)

    def __len__(self) -> int:
        return len(self.indices)

    def metadata_for_position(self, position: int) -> SampleMetadata:
        original_index = self.indices[position]
        original_class_id = self._targets[original_index]
        within_task_label = self.task_spec.within_task_label(original_class_id)
        return SampleMetadata(
            sample_id=self.sample_id_offset + original_index,
            task_id=self.task_spec.task_id,
            original_class_id=original_class_id,
            within_task_label=within_task_label,
            original_index=original_index,
            split=self.split,
        )

    def __getitem__(self, position: int) -> dict[str, Any]:
        metadata = self.metadata_for_position(position)
        item = self.dataset[metadata.original_index]
        return {
            "x": _extract_input(item),
            "y": metadata.within_task_label,
            "sample_id": metadata.sample_id,
            "task_id": metadata.task_id,
            "original_class_id": metadata.original_class_id,
            "within_task_label": metadata.within_task_label,
            "original_index": metadata.original_index,
            "split": metadata.split,
        }


class SplitCIFAR100TaskStream:
    """Class-incremental task stream for CIFAR-100-like datasets."""

    def __init__(
        self,
        *,
        dataset: ClassLabeledDataset,
        task_specs: Sequence[TaskSpec],
        split: str,
        sample_id_offset: int,
    ) -> None:
        self.dataset = dataset
        self.task_specs = tuple(task_specs)
        self.split = split
        self.sample_id_offset = int(sample_id_offset)
        self._targets = _as_int_targets(dataset.targets)
        self._indices_by_class = _indices_by_class(self._targets)
        self._task_datasets = tuple(
            self._build_task_dataset(task_spec) for task_spec in self.task_specs
        )

    @classmethod
    def from_dataset(
        cls,
        *,
        dataset: ClassLabeledDataset,
        task_count: int,
        classes_per_task: int,
        split_seed: int = 0,
        split: str = "train",
        sample_id_offset: int = 0,
        class_order: Sequence[int] | None = None,
    ) -> SplitCIFAR100TaskStream:
        """Build a task stream from any dataset exposing CIFAR-style targets."""

        task_specs = build_task_specs(
            task_count=task_count,
            classes_per_task=classes_per_task,
            split_seed=split_seed,
            targets=dataset.targets,
            class_order=class_order,
        )
        return cls(
            dataset=dataset,
            task_specs=task_specs,
            split=split,
            sample_id_offset=sample_id_offset,
        )

    def __len__(self) -> int:
        return len(self.task_specs)

    def __iter__(self) -> Iterator[TaskSubsetDataset]:
        return iter(self._task_datasets)

    def _build_task_dataset(self, task_spec: TaskSpec) -> TaskSubsetDataset:
        indices: list[int] = []
        for class_id in task_spec.class_ids:
            class_indices = self._indices_by_class.get(class_id)
            if not class_indices:
                raise ValueError(
                    f"dataset split {self.split!r} has no examples for class {class_id}"
                )
            indices.extend(class_indices)
        indices.sort()
        return TaskSubsetDataset(
            dataset=self.dataset,
            task_spec=task_spec,
            indices=indices,
            split=self.split,
            sample_id_offset=self.sample_id_offset,
        )

    def task_dataset(self, task_id: int) -> TaskSubsetDataset:
        """Return the dataset view for a task."""

        return self._task_datasets[task_id]


def load_cifar100_dataset(
    *,
    data_root: str | Path,
    train: bool,
    download: bool,
    transform: Any = None,
    target_transform: Any = None,
) -> ClassLabeledDataset:
    """Load torchvision CIFAR-100."""

    try:
        from torchvision.datasets import CIFAR100
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "torchvision is required to load the real CIFAR-100 dataset"
        ) from exc

    return CIFAR100(
        root=str(data_root),
        train=train,
        download=download,
        transform=transform,
        target_transform=target_transform,
    )


def build_split_cifar100_task_stream(
    *,
    config: SplitCIFAR100TaskStreamConfig = SplitCIFAR100TaskStreamConfig(),
    train: bool,
    transform: Any = None,
    target_transform: Any = None,
) -> SplitCIFAR100TaskStream:
    """Load CIFAR-100 and return a deterministic Split CIFAR-100 stream."""

    split = "train" if train else "test"
    sample_id_offset = (
        config.train_sample_id_offset if train else config.test_sample_id_offset
    )
    dataset = load_cifar100_dataset(
        data_root=config.data_root,
        train=train,
        download=config.download,
        transform=transform,
        target_transform=target_transform,
    )
    return SplitCIFAR100TaskStream.from_dataset(
        dataset=dataset,
        task_count=config.task_count,
        classes_per_task=config.classes_per_task,
        split_seed=config.split_seed,
        split=split,
        sample_id_offset=sample_id_offset,
    )
