"""Bounded replay buffers with stable sample metadata."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import torch


@dataclass
class ReplayItem:
    """One stored replay example plus metadata needed for analysis."""

    x: torch.Tensor
    target: int
    sample_id: int
    task_id: int
    original_class_id: int
    within_task_label: int
    original_index: int
    split: str
    added_at_task: int
    added_at_step: int
    replay_count: int = 0
    last_replayed_step: int | None = None


def _scalar(value: Any, index: int) -> int:
    if isinstance(value, torch.Tensor):
        return int(value[index].item())
    return int(value[index])


class ReservoirReplayBuffer:
    """Fixed-capacity reservoir buffer with uniform random replay sampling."""

    def __init__(self, *, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("replay buffer capacity must be positive")
        self.capacity = int(capacity)
        self._rng = Random(seed)
        self._items: list[ReplayItem] = []
        self.items_seen = 0
        self.total_replay_samples = 0
        self.replay_steps = 0

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> tuple[ReplayItem, ...]:
        return tuple(self._items)

    def add_batch(
        self,
        batch: dict[str, Any],
        *,
        target_key: str,
        added_at_task: int,
        added_at_step: int,
    ) -> None:
        """Insert a collated training batch using reservoir sampling."""

        if "x" not in batch:
            raise KeyError("batch is missing required key 'x'")
        batch_size = len(batch[target_key])
        x_batch = batch["x"]
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.as_tensor(x_batch)

        required_metadata = [
            "sample_id",
            "task_id",
            "original_class_id",
            "within_task_label",
            "original_index",
        ]
        for key in required_metadata:
            if key not in batch:
                raise KeyError(f"batch is missing replay metadata key {key!r}")

        for index in range(batch_size):
            item = ReplayItem(
                x=x_batch[index].detach().cpu().clone(),
                target=_scalar(batch[target_key], index),
                sample_id=_scalar(batch["sample_id"], index),
                task_id=_scalar(batch["task_id"], index),
                original_class_id=_scalar(batch["original_class_id"], index),
                within_task_label=_scalar(batch["within_task_label"], index),
                original_index=_scalar(batch["original_index"], index),
                split=str(batch["split"][index]) if "split" in batch else "train",
                added_at_task=added_at_task,
                added_at_step=added_at_step,
            )
            self._insert(item)

    def _insert(self, item: ReplayItem) -> None:
        self.items_seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return

        replacement_index = self._rng.randrange(self.items_seen)
        if replacement_index < self.capacity:
            self._items[replacement_index] = item

    def sample_batch(
        self,
        *,
        batch_size: int,
        target_key: str,
        replay_step: int,
    ) -> dict[str, torch.Tensor]:
        """Sample replay examples and update replay utilization counters."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")

        sample_count = min(batch_size, len(self._items))
        sampled_items = self._rng.sample(self._items, sample_count)
        return self._batch_from_items(
            sampled_items,
            target_key=target_key,
            replay_step=replay_step,
        )

    def sample_items(self, *, batch_size: int) -> list[ReplayItem]:
        """Sample replay items without updating replay utilization counters."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        sample_count = min(batch_size, len(self._items))
        return self._rng.sample(self._items, sample_count)

    def sample_batch_by_sample_ids(
        self,
        *,
        sample_ids: list[int],
        target_key: str,
        replay_step: int,
    ) -> dict[str, torch.Tensor]:
        """Return a replay batch for explicit sample IDs and update counters."""

        if not sample_ids:
            raise ValueError("sample_ids must not be empty")
        item_by_id = {item.sample_id: item for item in self._items}
        missing = [sample_id for sample_id in sample_ids if sample_id not in item_by_id]
        if missing:
            raise KeyError(f"sample IDs are not in replay buffer: {missing!r}")
        sampled_items = [item_by_id[sample_id] for sample_id in sample_ids]
        return self._batch_from_items(
            sampled_items,
            target_key=target_key,
            replay_step=replay_step,
        )

    def _batch_from_items(
        self,
        sampled_items: list[ReplayItem],
        *,
        target_key: str,
        replay_step: int,
    ) -> dict[str, torch.Tensor]:
        """Build a replay batch from selected items and update utilization."""

        sample_count = len(sampled_items)
        if sample_count < 1:
            raise ValueError("sampled_items must not be empty")
        self.replay_steps += 1
        self.total_replay_samples += sample_count

        for item in sampled_items:
            item.replay_count += 1
            item.last_replayed_step = replay_step

        return {
            "x": torch.stack([item.x for item in sampled_items]),
            target_key: torch.tensor([item.target for item in sampled_items], dtype=torch.long),
            "sample_id": torch.tensor([item.sample_id for item in sampled_items], dtype=torch.long),
            "task_id": torch.tensor([item.task_id for item in sampled_items], dtype=torch.long),
            "original_class_id": torch.tensor(
                [item.original_class_id for item in sampled_items],
                dtype=torch.long,
            ),
            "within_task_label": torch.tensor(
                [item.within_task_label for item in sampled_items],
                dtype=torch.long,
            ),
            "original_index": torch.tensor(
                [item.original_index for item in sampled_items],
                dtype=torch.long,
            ),
            "split": [item.split for item in sampled_items],
            "is_replay": torch.ones(sample_count, dtype=torch.bool),
            "replay_count": torch.tensor(
                [item.replay_count for item in sampled_items],
                dtype=torch.long,
            ),
            "last_replay_step": torch.tensor(
                [
                    item.last_replayed_step
                    if item.last_replayed_step is not None
                    else -1
                    for item in sampled_items
                ],
                dtype=torch.long,
            ),
        }

    def utilization_summary(self) -> dict[str, Any]:
        """Return replay utilization metrics for artifact logging."""

        replay_counts = [item.replay_count for item in self._items]
        task_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        for item in self._items:
            task_key = str(item.task_id)
            class_key = str(item.original_class_id)
            task_counts[task_key] = task_counts.get(task_key, 0) + 1
            class_counts[class_key] = class_counts.get(class_key, 0) + 1

        if replay_counts:
            mean_replay_count = sum(replay_counts) / len(replay_counts)
            max_replay_count = max(replay_counts)
            never_replayed_count = sum(1 for count in replay_counts if count == 0)
        else:
            mean_replay_count = 0.0
            max_replay_count = 0
            never_replayed_count = 0

        return {
            "capacity": self.capacity,
            "final_size": len(self._items),
            "items_seen": self.items_seen,
            "replay_steps": self.replay_steps,
            "total_replay_samples": self.total_replay_samples,
            "unique_replayed_samples": sum(1 for count in replay_counts if count > 0),
            "never_replayed_count": never_replayed_count,
            "mean_replay_count": mean_replay_count,
            "max_replay_count": max_replay_count,
            "buffer_task_counts": task_counts,
            "buffer_class_counts": class_counts,
        }
