import pytest
import torch

from src.models import build_mlp
from src.signals import (
    GRADIENT_SIGNAL_SCHEMA_VERSION,
    GradientSignalLogger,
    last_layer_gradient_signal_tensors,
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


def test_last_layer_gradient_signal_tensors_are_positive_and_batched():
    torch.manual_seed(0)
    model = build_mlp(input_shape=(2,), output_dim=4, hidden_dims=(3,))
    inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    targets = torch.tensor([0, 3])

    signals = last_layer_gradient_signal_tensors(
        model=model,
        inputs=inputs,
        targets=targets,
    )

    assert signals["logits"].shape == (2, 4)
    assert signals["last_layer_gradient_l2"].shape == (2,)
    assert torch.all(signals["last_layer_gradient_l2"] >= 0)
    assert torch.all(signals["classifier_weight_gradient_l2"] >= 0)


def test_gradient_signal_logger_records_eval_metadata():
    torch.manual_seed(0)
    model = build_mlp(input_shape=(2,), output_dim=4, hidden_dims=(3,))
    logger = GradientSignalLogger()

    logger.log_batch(
        model=model,
        inputs=torch.tensor([[1.0, -1.0], [0.5, 0.25]]),
        targets=torch.tensor([0, 3]),
        batch=_batch(),
        observation_type="seen_task_eval",
        trained_task_id=2,
        evaluated_task_id=1,
        global_step=12,
    )
    payload = logger.to_json_payload()
    row = payload["rows"][0]

    assert payload["schema_version"] == GRADIENT_SIGNAL_SCHEMA_VERSION
    assert row["sample_id"] == 10
    assert row["observation_type"] == "seen_task_eval"
    assert row["trained_task_id"] == 2
    assert row["evaluated_task_id"] == 1
    assert row["last_layer_gradient_l2"] == pytest.approx(
        (row["classifier_weight_gradient_l2"] ** 2 + row["classifier_bias_gradient_l2"] ** 2)
        ** 0.5
    )
    assert payload["summary"]["row_count"] == 2
