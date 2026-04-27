from src.experiments.mir_replay_comparison import (
    MIRReplayComparisonConfig,
    aggregate_mir_replay_comparison,
    baseline_config_for,
    load_config,
    planned_runs,
)


def _run(seed, final_accuracy, average_forgetting, replay_samples=10):
    return {
        "method_name": "mir_replay",
        "seed": seed,
        "run_name": f"mir_replay_{seed}",
        "run_dir": f"/tmp/mir_replay_{seed}",
        "metrics": {
            "final_accuracy": final_accuracy,
            "average_forgetting": average_forgetting,
            "training_time_seconds": 1.0 + seed,
            "method_metrics": {
                "replay": {
                    "total_replay_samples": replay_samples,
                    "unique_replayed_samples": 5 + seed,
                },
                "mir": {
                    "mean_selected_interference_score": 0.1 + seed,
                },
            },
        },
    }


def test_planned_mir_runs_use_seed_cross_product():
    config = MIRReplayComparisonConfig(
        comparison_name="unit",
        seeds=(0, 1),
        smoke=True,
    )

    plan = planned_runs(config)

    assert len(plan) == 2
    assert plan[0].method_name == "mir_replay"
    assert plan[0].run_name == "unit_mir_replay_seed0"
    assert plan[1].seed == 1


def test_baseline_config_for_matches_core_replay_budget():
    config = MIRReplayComparisonConfig(
        comparison_name="unit",
        smoke=True,
        replay_capacity=20,
        replay_batch_size=4,
        mir_candidate_size=8,
        mir_virtual_lr=0.2,
    )

    baseline_config = baseline_config_for(config, seed=7)

    assert baseline_config.method_name == "mir_replay"
    assert baseline_config.seed == 7
    assert baseline_config.replay_seed == 7
    assert baseline_config.replay_capacity == 20
    assert baseline_config.replay_batch_size == 4
    assert baseline_config.mir_candidate_size == 8
    assert baseline_config.mir_virtual_lr == 0.2


def test_aggregate_mir_replay_comparison_reports_means():
    config = MIRReplayComparisonConfig(
        comparison_name="unit",
        seeds=(0, 1),
        smoke=True,
    )
    runs = [
        _run(0, final_accuracy=0.1, average_forgetting=0.4, replay_samples=10),
        _run(1, final_accuracy=0.3, average_forgetting=0.2, replay_samples=20),
    ]

    summary = aggregate_mir_replay_comparison(config=config, runs=runs)

    metrics = summary["aggregates"]["mir_replay"]
    assert summary["fairness_controls"]["same_replay_sample_budget_as_random_replay"] is True
    assert metrics["final_accuracy_mean"] == 0.2
    assert metrics["average_forgetting_mean"] == 0.30000000000000004
    assert metrics["total_replay_samples_mean"] == 15
    assert metrics["unique_replayed_samples_mean"] == 5.5


def test_mir_replay_comparison_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/mir_replay_comparison_smoke.yaml")

    assert config.comparison_name == "task14_mir_replay_comparison_smoke"
    assert config.smoke is True
    assert config.seeds == (0, 1)
    assert config.hidden_dims == (32,)
    assert config.replay_capacity == 12
    assert config.mir_candidate_size == 8
