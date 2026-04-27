from pathlib import Path

from src.baselines.random_replay import (
    RandomReplayBaselineConfig,
    load_config,
    run_random_replay_baseline,
)
from src.experiment_tracking import load_experiment_artifacts


def test_random_replay_smoke_run_saves_replay_artifacts():
    output_root = Path(".tmp") / "test_random_replay_baseline"
    config = RandomReplayBaselineConfig(
        run_name="unit_random_replay_smoke",
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
        replay_capacity=12,
        replay_batch_size=4,
    )

    run = run_random_replay_baseline(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    replay_metrics = loaded["metrics"]["method_metrics"]["replay"]

    assert loaded["config"]["run"]["method_name"] == "random_replay"
    assert loaded["config"]["run"]["method"]["uses_replay"] is True
    assert loaded["config"]["run"]["replay"]["enabled"] is True
    assert loaded["config"]["run"]["signals"]["enabled"] is True
    assert replay_metrics["capacity"] == 12
    assert replay_metrics["final_size"] == 12
    assert replay_metrics["items_seen"] == 24
    assert replay_metrics["total_replay_samples"] > 0
    assert replay_metrics["replay_augmented_batch_count"] > 0
    assert 0.0 <= loaded["metrics"]["final_accuracy"] <= 1.0
    replay_signal_rows = [
        row for row in loaded["sample_signals"]["rows"] if row["observation_type"] == "replay_train"
    ]
    assert replay_signal_rows
    assert replay_signal_rows[0]["is_replay"] is True
    assert replay_signal_rows[0]["replay_count"] >= 1
    assert replay_signal_rows[0]["last_replay_step"] is not None


def test_random_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/random_replay_smoke.yaml")

    assert config.method_name == "random_replay"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.hidden_dims == (32,)
