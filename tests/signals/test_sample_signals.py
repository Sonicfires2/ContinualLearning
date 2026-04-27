import pytest
import torch

from src.signals import SIGNAL_SCHEMA_VERSION, SampleSignalLogger


def _batch():
    return {
        "sample_id": torch.tensor([10, 11]),
        "task_id": torch.tensor([1, 1]),
        "original_class_id": torch.tensor([2, 3]),
        "within_task_label": torch.tensor([0, 1]),
        "original_index": torch.tensor([100, 101]),
        "split": ["train", "train"],
        "is_replay": torch.tensor([False, True]),
        "replay_count": torch.tensor([0, 4]),
        "last_replay_step": torch.tensor([-1, 12]),
    }


def test_sample_signal_logger_records_joinable_metadata_and_uncertainty():
    logger = SampleSignalLogger()

    logger.log_batch(
        logits=torch.tensor([[2.0, 0.0, -1.0, -2.0], [0.0, 0.1, 0.2, 2.0]]),
        targets=torch.tensor([0, 3]),
        batch=_batch(),
        observation_type="replay_train",
        trained_task_id=2,
        epoch=0,
        global_step=12,
    )

    payload = logger.to_json_payload()
    first_row = payload["rows"][0]
    second_row = payload["rows"][1]

    assert payload["schema_version"] == SIGNAL_SCHEMA_VERSION
    assert first_row["sample_id"] == 10
    assert first_row["source_task_id"] == 1
    assert first_row["original_class_id"] == 2
    assert first_row["split"] == "train"
    assert first_row["correct"] is True
    assert first_row["confidence"] == pytest.approx(0.83095264)
    assert first_row["uncertainty"] == pytest.approx(1.0 - first_row["confidence"])
    assert second_row["is_replay"] is True
    assert second_row["replay_count"] == 4
    assert second_row["last_replay_step"] == 12
    assert payload["summary"]["row_count"] == 2
    assert payload["summary"]["counts_by_observation_type"] == {"replay_train": 2}


def test_sample_signal_logger_fails_without_stable_sample_metadata():
    logger = SampleSignalLogger()

    with pytest.raises(KeyError, match="sample_id"):
        logger.log_batch(
            logits=torch.tensor([[1.0, 0.0]]),
            targets=torch.tensor([0]),
            batch={"task_id": torch.tensor([0])},
            observation_type="current_train",
            trained_task_id=0,
            global_step=0,
        )
