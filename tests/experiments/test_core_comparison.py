from src.experiments.core_comparison import (
    CORE_METHODS,
    CoreComparisonConfig,
    aggregate_core_comparison,
    baseline_config_for,
    load_config,
    planned_runs,
)


def _run(method_name, seed, final_accuracy, average_forgetting, replay_samples):
    return {
        "method_name": method_name,
        "seed": seed,
        "run_name": f"{method_name}_{seed}",
        "run_dir": f"/tmp/{method_name}_{seed}",
        "metrics": {
            "final_accuracy": final_accuracy,
            "average_forgetting": average_forgetting,
            "training_time_seconds": 1.0 + seed,
            "method_metrics": {
                "replay": {
                    "total_replay_samples": replay_samples,
                }
            },
        },
    }


def test_planned_runs_are_method_seed_cross_product():
    config = CoreComparisonConfig(
        comparison_name="unit",
        seeds=(0, 1),
        smoke=True,
    )

    plan = planned_runs(config)

    assert len(plan) == 2 * len(CORE_METHODS)
    assert plan[0].method_name == "fine_tuning"
    assert plan[0].seed == 0
    assert plan[-1].method_name == "spaced_replay"
    assert plan[-1].seed == 1


def test_baseline_config_for_uses_budget_matched_fixed_periodic_replay():
    config = CoreComparisonConfig(
        comparison_name="unit",
        smoke=True,
        fixed_periodic_replay_interval=1,
        fixed_periodic_budget_mode="budget_matched",
    )

    fixed = baseline_config_for(config, method_name="fixed_periodic_replay", seed=7)
    spaced = baseline_config_for(config, method_name="spaced_replay", seed=7)

    assert fixed.seed == 7
    assert fixed.replay_seed == 7
    assert fixed.replay_interval == 1
    assert fixed.budget_mode == "budget_matched"
    assert spaced.scheduler_budget_mode == "match_random_replay"
    assert spaced.replay_seed == 7


def test_aggregate_core_comparison_reports_method_means():
    config = CoreComparisonConfig(comparison_name="unit", seeds=(0, 1), smoke=True)
    runs = []
    for method_name in CORE_METHODS:
        runs.append(_run(method_name, 0, 0.1, 0.4, 0 if method_name == "fine_tuning" else 10))
        runs.append(_run(method_name, 1, 0.3, 0.2, 0 if method_name == "fine_tuning" else 20))

    summary = aggregate_core_comparison(config=config, runs=runs)

    assert summary["fairness_controls"]["budget_matched_replay_methods"] is True
    assert summary["aggregates"]["random_replay"]["final_accuracy_mean"] == 0.2
    assert summary["aggregates"]["random_replay"]["total_replay_samples_mean"] == 15
    assert summary["aggregates"]["fine_tuning"]["total_replay_samples_mean"] == 0


def test_core_comparison_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/core_comparison_smoke.yaml")

    assert config.comparison_name == "task13_core_comparison_smoke"
    assert config.smoke is True
    assert config.seeds == (0, 1)
    assert config.hidden_dims == (32,)
    assert config.fixed_periodic_replay_interval == 1
