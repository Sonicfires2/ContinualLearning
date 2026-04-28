from pathlib import Path

from src.baselines.mir_interference_diagnostic import (
    MIRInterferenceDiagnosticConfig,
    load_config,
    run_mir_interference_diagnostic,
)
from src.experiment_tracking import load_experiment_artifacts


def test_mir_interference_diagnostic_smoke_saves_candidate_report():
    output_root = Path(".tmp") / "test_mir_interference_diagnostic"
    config = MIRInterferenceDiagnosticConfig(
        run_name="unit_mir_interference_diagnostic_smoke",
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
        min_replay_interval_steps=1,
        max_replay_interval_steps=8,
    )

    run = run_mir_interference_diagnostic(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    diagnostic = loaded["mir_interference_diagnostic"]
    summary = diagnostic["summary"]

    assert loaded["config"]["run"]["method_name"] == "mir_interference_diagnostic"
    assert loaded["config"]["run"]["method"]["diagnostic_only"] is True
    assert summary["candidate_row_count"] > 0
    assert summary["learned_risk_predicts_mir_topk"]["average_precision"] is not None
    assert summary["event_topk_overlap"]["mean_topk_overlap"] is not None
    assert loaded["metrics"]["method_metrics"]["mir_interference"] == summary


def test_mir_interference_diagnostic_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/mir_interference_diagnostic_smoke.yaml")

    assert config.method_name == "mir_interference_diagnostic"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.mir_candidate_size == 8
    assert config.mir_virtual_lr == 0.2
    assert config.hidden_dims == (32,)

