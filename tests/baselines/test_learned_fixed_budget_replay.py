from pathlib import Path

from src.baselines.learned_fixed_budget_replay import (
    LearnedFixedBudgetReplayConfig,
    load_config,
    run_learned_fixed_budget_replay,
)
from src.experiment_tracking import load_experiment_artifacts


def test_learned_fixed_budget_replay_smoke_fills_replay_budget_without_skips():
    output_root = Path(".tmp") / "test_learned_fixed_budget_replay"
    config = LearnedFixedBudgetReplayConfig(
        run_name="unit_learned_fixed_budget_replay_smoke",
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
        scheduler_budget_mode="risk_ranked",
    )

    run = run_learned_fixed_budget_replay(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    replay_metrics = loaded["metrics"]["method_metrics"]["replay"]
    scheduler_metrics = loaded["metrics"]["method_metrics"]["scheduler"]

    assert loaded["config"]["run"]["method_name"] == "learned_fixed_budget_replay"
    assert loaded["config"]["run"]["method"]["matched_replay_budget_to_random_replay"]
    assert loaded["config"]["run"]["replay"]["schedule"] == "learned_fixed_budget_replay"
    assert loaded["config"]["run"]["replay"]["scheduler"]["budget_mode"] == "risk_ranked"
    assert loaded["config"]["run"]["replay"]["scheduler"]["risk_score_source"] == "learned"
    assert scheduler_metrics["budget_mode"] == "risk_ranked"
    assert scheduler_metrics["skipped_selection_event_count"] == 0
    assert scheduler_metrics["trace_row_count"] == replay_metrics["total_replay_samples"]
    assert replay_metrics["total_replay_samples"] == (
        replay_metrics["replay_augmented_batch_count"] * config.replay_batch_size
    )


def test_learned_fixed_budget_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/learned_fixed_budget_replay_smoke.yaml")

    assert config.method_name == "learned_fixed_budget_replay"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.scheduler_budget_mode == "risk_ranked"
    assert config.learned_predictor_feature_group == "all_features"
    assert config.hidden_dims == (32,)
