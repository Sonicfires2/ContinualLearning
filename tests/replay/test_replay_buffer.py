import torch

from src.replay import ReservoirReplayBuffer


def _batch(sample_ids):
    return {
        "x": torch.stack([torch.tensor([float(sample_id)]) for sample_id in sample_ids]),
        "original_class_id": torch.tensor(sample_ids, dtype=torch.long),
        "sample_id": torch.tensor(sample_ids, dtype=torch.long),
        "task_id": torch.zeros(len(sample_ids), dtype=torch.long),
        "within_task_label": torch.tensor(sample_ids, dtype=torch.long),
        "original_index": torch.tensor(sample_ids, dtype=torch.long),
        "split": ["train" for _ in sample_ids],
    }


def test_reservoir_buffer_enforces_capacity_and_tracks_seen_items():
    buffer = ReservoirReplayBuffer(capacity=3, seed=7)

    buffer.add_batch(
        _batch([0, 1, 2, 3, 4]),
        target_key="original_class_id",
        added_at_task=0,
        added_at_step=0,
    )

    assert len(buffer) == 3
    assert buffer.items_seen == 5


def test_replay_sampling_updates_utilization_counts():
    buffer = ReservoirReplayBuffer(capacity=4, seed=1)
    buffer.add_batch(
        _batch([0, 1, 2, 3]),
        target_key="original_class_id",
        added_at_task=0,
        added_at_step=0,
    )

    replay_batch = buffer.sample_batch(
        batch_size=2,
        target_key="original_class_id",
        replay_step=10,
    )
    summary = buffer.utilization_summary()

    assert replay_batch["x"].shape == (2, 1)
    assert replay_batch["original_class_id"].shape == (2,)
    assert replay_batch["is_replay"].tolist() == [True, True]
    assert replay_batch["replay_count"].tolist() == [1, 1]
    assert replay_batch["last_replay_step"].tolist() == [10, 10]
    assert len(replay_batch["split"]) == 2
    assert summary["replay_steps"] == 1
    assert summary["total_replay_samples"] == 2
    assert summary["unique_replayed_samples"] == 2
    assert summary["never_replayed_count"] == 2


def test_replay_sampling_by_sample_ids_preserves_selection_order():
    buffer = ReservoirReplayBuffer(capacity=4, seed=1)
    buffer.add_batch(
        _batch([0, 1, 2, 3]),
        target_key="original_class_id",
        added_at_task=0,
        added_at_step=0,
    )

    replay_batch = buffer.sample_batch_by_sample_ids(
        sample_ids=[3, 1],
        target_key="original_class_id",
        replay_step=7,
    )
    summary = buffer.utilization_summary()

    assert replay_batch["sample_id"].tolist() == [3, 1]
    assert replay_batch["last_replay_step"].tolist() == [7, 7]
    assert summary["replay_steps"] == 1
    assert summary["total_replay_samples"] == 2


def test_candidate_item_sampling_does_not_update_replay_utilization():
    buffer = ReservoirReplayBuffer(capacity=4, seed=1)
    buffer.add_batch(
        _batch([0, 1, 2, 3]),
        target_key="original_class_id",
        added_at_task=0,
        added_at_step=0,
    )

    candidates = buffer.sample_items(batch_size=3)
    summary = buffer.utilization_summary()

    assert len(candidates) == 3
    assert summary["replay_steps"] == 0
    assert summary["total_replay_samples"] == 0
    assert summary["unique_replayed_samples"] == 0
