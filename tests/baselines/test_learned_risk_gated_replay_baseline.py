from pathlib import Path

from src.baselines.learned_risk_gated_replay import (
    LearnedRiskGatedReplayBaselineConfig,
    load_config,
    run_learned_risk_gated_replay_baseline,
)
from src.experiment_tracking import load_experiment_artifacts


def test_learned_risk_gated_replay_smoke_run_saves_learned_gate_metadata():
    output_root = Path(".tmp") / "test_learned_risk_gated_replay_baseline"
    config = LearnedRiskGatedReplayBaselineConfig(
        run_name="unit_learned_risk_gated_replay_smoke",
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
        risk_threshold=0.9998,
        scheduler_budget_mode="risk_only",
    )

    run = run_learned_risk_gated_replay_baseline(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    replay_metrics = loaded["metrics"]["method_metrics"]["replay"]
    scheduler_metrics = loaded["metrics"]["method_metrics"]["scheduler"]

    assert loaded["config"]["run"]["method_name"] == "learned_risk_gated_replay"
    assert loaded["config"]["run"]["method"]["learned_predictor_gate"] is True
    assert loaded["config"]["run"]["replay"]["schedule"] == "learned_risk_gated_replay"
    assert loaded["config"]["run"]["replay"]["scheduler"]["risk_score_source"] == "learned"
    assert loaded["config"]["run"]["replay"]["risk_scorer"]["feature_group"] == "all_features"
    assert scheduler_metrics["risk_score_source"] == "learned"
    assert scheduler_metrics["skipped_selection_event_count"] > 0
    assert loaded["scheduler_trace"]["skipped_rows"]
    assert replay_metrics["total_replay_samples"] > 0


def test_learned_risk_gated_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/learned_risk_gated_replay_smoke.yaml")

    assert config.method_name == "learned_risk_gated_replay"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.risk_threshold == 0.9998
    assert config.scheduler_budget_mode == "risk_only"
    assert config.learned_predictor_feature_group == "all_features"
    assert config.hidden_dims == (32,)
