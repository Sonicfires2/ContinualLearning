from pathlib import Path

import pytest

from src.baselines.fixed_periodic_replay import (
    FixedPeriodicReplayBaselineConfig,
    _should_replay_on_step,
    load_config,
    run_fixed_periodic_replay_baseline,
)
from src.experiment_tracking import load_experiment_artifacts


def test_periodic_replay_decision_uses_one_indexed_optimizer_steps():
    assert _should_replay_on_step(global_step=0, replay_interval=2) is False
    assert _should_replay_on_step(global_step=1, replay_interval=2) is True
    assert _should_replay_on_step(global_step=2, replay_interval=2) is False
    assert _should_replay_on_step(global_step=3, replay_interval=2) is True

    with pytest.raises(ValueError, match="replay_interval"):
        _should_replay_on_step(global_step=0, replay_interval=0)


def test_fixed_periodic_replay_smoke_run_saves_cadence_artifacts():
    output_root = Path(".tmp") / "test_fixed_periodic_replay_baseline"
    config = FixedPeriodicReplayBaselineConfig(
        run_name="unit_fixed_periodic_replay_smoke",
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
        replay_interval=2,
    )

    run = run_fixed_periodic_replay_baseline(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    replay_metrics = loaded["metrics"]["method_metrics"]["replay"]

    assert loaded["config"]["run"]["method_name"] == "fixed_periodic_replay"
    assert loaded["config"]["run"]["method"]["uses_replay"] is True
    assert loaded["config"]["run"]["method"]["forgetting_aware"] is False
    assert loaded["config"]["run"]["method"]["estimates_t_i"] is False
    assert loaded["config"]["run"]["replay"]["schedule"] == "fixed_periodic"
    assert loaded["config"]["run"]["replay"]["replay_interval"] == 2
    assert replay_metrics["schedule"] == "fixed_periodic"
    assert replay_metrics["budget_mode"] == "interval_ablation"
    assert replay_metrics["replay_interval"] == 2
    assert replay_metrics["capacity"] == 12
    assert replay_metrics["final_size"] == 12
    assert replay_metrics["items_seen"] == 24
    assert replay_metrics["total_replay_samples"] > 0
    assert replay_metrics["replay_augmented_batch_count"] == len(
        replay_metrics["replay_event_steps"]
    )
    assert replay_metrics["skipped_replay_steps_due_to_empty_buffer"] > 0
    assert replay_metrics["skipped_replay_steps_due_to_interval"] > 0
    assert all(
        optimizer_step % 2 == 0
        for optimizer_step in replay_metrics["replay_event_optimizer_steps"]
    )

    replay_signal_rows = [
        row for row in loaded["sample_signals"]["rows"] if row["observation_type"] == "replay_train"
    ]
    assert replay_signal_rows
    assert {row["global_step"] for row in replay_signal_rows}.issubset(
        set(replay_metrics["replay_event_steps"])
    )
    assert all(row["source_task_id"] < row["trained_task_id"] for row in replay_signal_rows)


def test_fixed_periodic_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/fixed_periodic_replay_smoke.yaml")

    assert config.method_name == "fixed_periodic_replay"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.replay_interval == 2
    assert config.budget_mode == "interval_ablation"
    assert config.hidden_dims == (32,)
