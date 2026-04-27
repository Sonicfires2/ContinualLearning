from pathlib import Path

from src.baselines.spaced_replay import (
    SpacedReplayBaselineConfig,
    load_config,
    run_spaced_replay_baseline,
)
from src.experiment_tracking import load_experiment_artifacts


def test_spaced_replay_smoke_run_saves_scheduler_trace():
    output_root = Path(".tmp") / "test_spaced_replay_baseline"
    config = SpacedReplayBaselineConfig(
        run_name="unit_spaced_replay_smoke",
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
        min_replay_interval_steps=1,
        max_replay_interval_steps=8,
    )

    run = run_spaced_replay_baseline(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    replay_metrics = loaded["metrics"]["method_metrics"]["replay"]
    scheduler_metrics = loaded["metrics"]["method_metrics"]["scheduler"]

    assert loaded["config"]["run"]["method_name"] == "spaced_replay"
    assert loaded["config"]["run"]["method"]["uses_replay"] is True
    assert loaded["config"]["run"]["method"]["forgetting_aware"] is True
    assert loaded["config"]["run"]["method"]["estimates_t_i"] is True
    assert loaded["config"]["run"]["replay"]["schedule"] == "spaced_replay"
    assert loaded["scheduler_trace"]["summary"]["trace_row_count"] > 0
    assert scheduler_metrics["trace_row_count"] == loaded["scheduler_trace"]["summary"]["trace_row_count"]
    assert scheduler_metrics["budget_mode"] == "match_random_replay"
    assert replay_metrics["total_replay_samples"] > 0
    assert replay_metrics["replay_augmented_batch_count"] > 0

    trace_rows = loaded["scheduler_trace"]["rows"]
    replay_signal_rows = [
        row for row in loaded["sample_signals"]["rows"] if row["observation_type"] == "replay_train"
    ]
    assert trace_rows
    assert replay_signal_rows
    assert {row["global_step"] for row in replay_signal_rows}.issubset(
        {row["global_step"] for row in trace_rows}
    )
    assert all(row["source_task_id"] < row["trained_task_id"] for row in replay_signal_rows)


def test_spaced_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/spaced_replay_smoke.yaml")

    assert config.method_name == "spaced_replay"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.min_replay_interval_steps == 1
    assert config.max_replay_interval_steps == 8
    assert config.scheduler_budget_mode == "match_random_replay"
    assert config.hidden_dims == (32,)
