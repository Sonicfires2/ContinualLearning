from pathlib import Path
from uuid import uuid4

import pytest

from src.baselines.fine_tuning import (
    BaselineDataUnavailableError,
    FineTuningBaselineConfig,
    load_config,
    run_fine_tuning_baseline,
)
from src.experiment_tracking import load_experiment_artifacts


def test_fine_tuning_smoke_run_saves_complete_artifacts():
    output_root = Path(".tmp") / "test_fine_tuning_baseline"
    config = FineTuningBaselineConfig(
        run_name="unit_fine_tuning_smoke",
        output_root=str(output_root),
        overwrite=True,
        smoke=True,
        task_count=3,
        classes_per_task=2,
        epochs_per_task=3,
        batch_size=8,
        eval_batch_size=16,
        learning_rate=0.2,
        device="cpu",
        hidden_dims=(32,),
    )

    run = run_fine_tuning_baseline(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)

    assert run.result.task_count == 3
    assert loaded["config"]["run"]["method_name"] == "fine_tuning"
    assert loaded["config"]["run"]["dataset"]["smoke"] is True
    assert loaded["config"]["run"]["method"]["uses_replay"] is False
    assert loaded["config"]["run"]["replay"]["enabled"] is False
    assert loaded["config"]["run"]["signals"]["enabled"] is True
    assert loaded["config"]["run"]["dataset"]["target_key"] == "original_class_id"
    assert loaded["metrics"]["task_count"] == 3
    assert 0.0 <= loaded["metrics"]["final_accuracy"] <= 1.0
    assert loaded["metrics"]["average_forgetting"] >= 0.0
    assert loaded["sample_signals"]["summary"]["row_count"] > 0
    assert loaded["sample_signals"]["rows"][0]["observation_type"] in {
        "current_train",
        "seen_task_eval",
    }
    assert {"sample_id", "source_task_id", "original_class_id", "split"}.issubset(
        loaded["sample_signals"]["rows"][0]
    )


def test_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/fine_tuning_smoke.yaml")

    assert config.method_name == "fine_tuning"
    assert config.smoke is True
    assert config.target_key == "original_class_id"
    assert config.hidden_dims == (32,)


def test_real_baseline_fails_cleanly_when_dataset_is_missing():
    config = FineTuningBaselineConfig(
        run_name="missing-data",
        output_root=str(Path(".tmp") / "missing_data_baseline"),
        smoke=False,
        data_root=str(Path(".tmp") / "missing_cifar100" / uuid4().hex),
        download=False,
        task_count=10,
        classes_per_task=10,
    )

    with pytest.raises(BaselineDataUnavailableError, match="Split CIFAR-100"):
        run_fine_tuning_baseline(config)
