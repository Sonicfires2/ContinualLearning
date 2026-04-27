from pathlib import Path

import torch

from src.baselines.learned_hybrid_replay import (
    LearnedHybridReplayConfig,
    _select_diversity_items,
    load_config,
    run_learned_hybrid_replay,
)
from src.experiment_tracking import load_experiment_artifacts
from src.replay import ReplayItem


def _item(sample_id: int, class_id: int) -> ReplayItem:
    return ReplayItem(
        x=torch.tensor([float(sample_id)]),
        target=class_id,
        sample_id=sample_id,
        task_id=0,
        original_class_id=class_id,
        within_task_label=0,
        original_index=sample_id,
        split="train",
        added_at_task=0,
        added_at_step=0,
    )


def test_class_balanced_diversity_selector_spreads_classes():
    from random import Random

    items = (
        _item(1, 0),
        _item(2, 0),
        _item(3, 0),
        _item(4, 1),
        _item(5, 1),
        _item(6, 2),
    )

    selected = _select_diversity_items(
        items=items,
        excluded_sample_ids={1},
        quota=3,
        mode="class_balanced",
        rng=Random(0),
    )

    assert len(selected) == 3
    assert 1 not in {item.sample_id for item in selected}
    assert len({item.original_class_id for item in selected}) == 3


def test_learned_hybrid_replay_smoke_matches_budget_and_logs_mix():
    output_root = Path(".tmp") / "test_learned_hybrid_replay"
    config = LearnedHybridReplayConfig(
        run_name="unit_learned_hybrid_replay_smoke",
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
        learned_risk_fraction=0.5,
        hybrid_diversity_mode="class_balanced",
    )

    run = run_learned_hybrid_replay(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    replay_metrics = loaded["metrics"]["method_metrics"]["replay"]
    scheduler_metrics = loaded["metrics"]["method_metrics"]["scheduler"]
    hybrid_metrics = replay_metrics["hybrid"]

    assert loaded["config"]["run"]["method_name"] == "learned_hybrid_replay"
    assert loaded["config"]["run"]["method"]["protects_replay_diversity"]
    assert loaded["config"]["run"]["replay"]["schedule"] == "learned_hybrid_replay"
    assert loaded["config"]["run"]["replay"]["hybrid_diversity_mode"] == "class_balanced"
    assert replay_metrics["total_replay_samples"] == (
        replay_metrics["replay_augmented_batch_count"] * config.replay_batch_size
    )
    assert scheduler_metrics["skipped_selection_event_count"] == 0
    assert hybrid_metrics["learned_risk_selected_count"] > 0
    assert hybrid_metrics["diversity_selected_count"] > 0
    assert hybrid_metrics["actual_learned_risk_fraction"] == 0.5


def test_learned_hybrid_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/learned_hybrid_replay_smoke.yaml")

    assert config.method_name == "learned_hybrid_replay"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.learned_risk_fraction == 0.5
    assert config.hybrid_diversity_mode == "class_balanced"
    assert config.hidden_dims == (32,)


def test_task22_rescue_ablation_configs_are_runnable_specs():
    config_specs = {
        "configs/experiments/learned_hybrid_replay_task22_frac025_class_balanced_split_cifar100.yaml": (
            0.25,
            "class_balanced",
        ),
        "configs/experiments/learned_hybrid_replay_task22_frac025_random_split_cifar100.yaml": (
            0.25,
            "random",
        ),
        "configs/experiments/learned_hybrid_replay_task22_class_balanced_only_split_cifar100.yaml": (
            0.0,
            "class_balanced",
        ),
    }

    for config_path, (expected_fraction, expected_mode) in config_specs.items():
        config = load_config(config_path)

        assert config.protocol_id == "core_split_cifar100_v2"
        assert config.smoke is False
        assert config.replay_batch_size == 32
        assert config.learned_risk_fraction == expected_fraction
        assert config.hybrid_diversity_mode == expected_mode
        assert config.output_root == "experiments/runs/task22_rescue_ablation"
