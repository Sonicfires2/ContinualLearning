"""Data utilities for continual-learning task streams."""

from src.data.split_cifar100 import (
    DEFAULT_SAMPLE_ID_OFFSETS,
    SampleMetadata,
    SplitCIFAR100TaskStream,
    SplitCIFAR100TaskStreamConfig,
    TaskSpec,
    TaskSubsetDataset,
    build_split_cifar100_task_stream,
    build_task_specs,
    load_cifar100_dataset,
)

__all__ = [
    "DEFAULT_SAMPLE_ID_OFFSETS",
    "SampleMetadata",
    "SplitCIFAR100TaskStream",
    "SplitCIFAR100TaskStreamConfig",
    "TaskSpec",
    "TaskSubsetDataset",
    "build_split_cifar100_task_stream",
    "build_task_specs",
    "load_cifar100_dataset",
]
