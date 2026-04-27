from pathlib import Path

from src.baselines.mir_replay import (
    MIRReplayBaselineConfig,
    load_config,
    run_mir_replay_baseline,
)
from src.experiment_tracking import load_experiment_artifacts


def test_mir_replay_smoke_run_saves_mir_trace():
    output_root = Path(".tmp") / "test_mir_replay_baseline"
    config = MIRReplayBaselineConfig(
        run_name="unit_mir_replay_smoke",
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
        mir_candidate_size=8,
        mir_virtual_lr=0.2,
    )

    run = run_mir_replay_baseline(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    replay_metrics = loaded["metrics"]["method_metrics"]["replay"]
    mir_metrics = loaded["metrics"]["method_metrics"]["mir"]

    assert loaded["config"]["run"]["method_name"] == "mir_replay"
    assert loaded["config"]["run"]["method"]["uses_replay"] is True
    assert loaded["config"]["run"]["method"]["selection_policy"] == "er_mir_mi1"
    assert loaded["config"]["run"]["replay"]["sampling_policy"] == "maximally_interfered_retrieval"
    assert loaded["mir_trace"]["summary"]["trace_row_count"] > 0
    assert mir_metrics["trace_row_count"] == loaded["mir_trace"]["summary"]["trace_row_count"]
    assert replay_metrics["total_replay_samples"] > 0
    assert replay_metrics["replay_augmented_batch_count"] > 0
    assert replay_metrics["mir_candidate_size"] == 8

    trace_rows = loaded["mir_trace"]["rows"]
    replay_signal_rows = [
        row for row in loaded["sample_signals"]["rows"] if row["observation_type"] == "replay_train"
    ]
    assert trace_rows
    assert replay_signal_rows
    assert {row["global_step"] for row in replay_signal_rows}.issubset(
        {row["global_step"] for row in trace_rows}
    )
    assert all(row["source_task_id"] < row["trained_task_id"] for row in replay_signal_rows)


def test_mir_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/mir_replay_smoke.yaml")

    assert config.method_name == "mir_replay"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.mir_candidate_size == 8
    assert config.mir_virtual_lr == 0.2
    assert config.hidden_dims == (32,)
