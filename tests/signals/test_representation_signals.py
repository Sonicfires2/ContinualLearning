import pytest
import torch

from src.models import build_mlp
from src.signals import (
    REPRESENTATION_SIGNAL_SCHEMA_VERSION,
    RepresentationDriftLogger,
    representation_signal_tensors,
)


def _batch():
    return {
        "sample_id": torch.tensor([10, 11]),
        "task_id": torch.tensor([1, 1]),
        "original_class_id": torch.tensor([2, 3]),
        "within_task_label": torch.tensor([0, 1]),
        "original_index": torch.tensor([100, 101]),
        "split": ["test", "test"],
    }


def test_representation_signal_tensors_are_batched():
    torch.manual_seed(0)
    model = build_mlp(input_shape=(2,), output_dim=4, hidden_dims=(3,))
    inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]])

    signals = representation_signal_tensors(model=model, inputs=inputs)

    assert signals["logits"].shape == (2, 4)
    assert signals["representations"].shape == (2, 3)
    assert signals["representation_l2"].shape == (2,)
    assert torch.all(signals["representation_l2"] >= 0)


def test_representation_drift_logger_uses_first_observation_as_reference():
    torch.manual_seed(0)
    model = build_mlp(input_shape=(2,), output_dim=4, hidden_dims=(3,))
    logger = RepresentationDriftLogger()
    inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    targets = torch.tensor([0, 3])

    logger.log_batch(
        model=model,
        inputs=inputs,
        targets=targets,
        batch=_batch(),
        observation_type="seen_task_eval",
        trained_task_id=1,
        evaluated_task_id=1,
        global_step=10,
    )
    logger.log_batch(
        model=model,
        inputs=inputs + 0.2,
        targets=targets,
        batch=_batch(),
        observation_type="seen_task_eval",
        trained_task_id=2,
        evaluated_task_id=1,
        global_step=20,
    )
    payload = logger.to_json_payload()
    first_row = payload["rows"][0]
    second_observation = payload["rows"][2]

    assert payload["schema_version"] == REPRESENTATION_SIGNAL_SCHEMA_VERSION
    assert first_row["representation_drift"] == pytest.approx(0.0)
    assert second_observation["reference_trained_task_id"] == 1
    assert second_observation["reference_global_step"] == 10
    assert second_observation["representation_drift"] >= 0.0
    assert payload["summary"]["row_count"] == 4
