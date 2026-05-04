import pytest

from src.baselines.representation_drift_replay import (
    RepresentationDriftReplayConfig,
    load_config,
    run_representation_drift_replay,
)
from src.experiment_tracking import load_experiment_artifacts


@pytest.mark.parametrize(
    "variant",
    ["drift_ranked", "drift_due_time", "drift_hybrid"],
)
def test_representation_drift_replay_smoke_records_scheduler_trace(variant):
    run = run_representation_drift_replay(
        RepresentationDriftReplayConfig(
            run_name=f"representation_drift_replay_{variant}_test_smoke",
            output_root=".tmp/test_representation_drift_replay",
            overwrite=True,
            smoke=True,
            task_count=3,
            classes_per_task=2,
            epochs_per_task=1,
            batch_size=8,
            eval_batch_size=16,
            learning_rate=0.2,
            device="cpu",
            hidden_dims=(16,),
            replay_capacity=12,
            replay_batch_size=4,
            log_signals=True,
            log_representation_signals=True,
            drift_variant=variant,
            drift_candidate_count=8,
            drift_scoring_batch_size=8,
            max_replay_interval_steps=8,
        )
    )
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    metrics = loaded["metrics"]["method_metrics"]

    assert loaded["config"]["run"]["replay"]["sampling_policy"] == variant
    assert metrics["drift_scheduler"]["variant"] == variant
    assert metrics["drift_scheduler"]["trace_row_count"] > 0
    assert metrics["replay"]["total_replay_samples"] > 0
    assert metrics["representation_signals"]["row_count"] > 0
    assert loaded["scheduler_trace"]["variant"] == variant
    assert loaded["scheduler_trace"]["records"]
    assert (run.artifacts.run_dir / "scheduler_trace.json").exists()


def test_representation_drift_replay_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/representation_drift_replay_smoke.yaml")

    assert config.method_name == "representation_drift_replay"
    assert config.smoke is True
    assert config.drift_variant == "drift_ranked"
    assert config.drift_candidate_count == 8
    assert config.hidden_dims == (32,)
