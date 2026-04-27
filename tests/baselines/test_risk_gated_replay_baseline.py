from pathlib import Path

from src.baselines.risk_gated_replay import (
    RiskGatedReplayBaselineConfig,
    load_config,
    run_risk_gated_replay_baseline,
)
from src.experiment_tracking import load_experiment_artifacts


def test_risk_gated_replay_smoke_run_saves_skip_aware_scheduler_trace():
    output_root = Path(".tmp") / "test_risk_gated_replay_baseline"
    config = RiskGatedReplayBaselineConfig(
        run_name="unit_risk_gated_replay_smoke",
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
        risk_threshold=0.5,
        scheduler_budget_mode="risk_and_due",
    )

    run = run_risk_gated_replay_baseline(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    replay_metrics = loaded["metrics"]["method_metrics"]["replay"]
    scheduler_metrics = loaded["metrics"]["method_metrics"]["scheduler"]

    assert loaded["config"]["run"]["method_name"] == "risk_gated_replay"
    assert loaded["config"]["run"]["method"]["event_triggered"] is True
    assert loaded["config"]["run"]["method"]["skips_low_risk_replay"] is True
    assert loaded["config"]["run"]["replay"]["schedule"] == "risk_gated_replay"
    assert scheduler_metrics["budget_mode"] == "risk_and_due"
    assert scheduler_metrics["risk_threshold"] == 0.5
    assert scheduler_metrics["skipped_selection_event_count"] > 0
    assert loaded["scheduler_trace"]["skipped_rows"]
    assert replay_metrics["total_replay_samples"] > 0
    assert replay_metrics["total_replay_samples"] < (
        replay_metrics["current_batch_count"] * config.replay_batch_size
    )


def test_risk_gated_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/risk_gated_replay_smoke.yaml")

    assert config.method_name == "risk_gated_replay"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.min_replay_interval_steps == 1
    assert config.max_replay_interval_steps == 8
    assert config.risk_threshold == 0.5
    assert config.scheduler_budget_mode == "risk_and_due"
    assert config.hidden_dims == (32,)
