from pathlib import Path

from src.baselines.gradient_signal_diagnostic import (
    GradientSignalDiagnosticConfig,
    load_config,
    run_gradient_signal_diagnostic,
)
from src.experiment_tracking import load_experiment_artifacts


def test_gradient_signal_diagnostic_smoke_writes_gradient_artifact():
    output_root = Path(".tmp") / "test_gradient_signal_diagnostic"
    config = GradientSignalDiagnosticConfig(
        run_name="unit_gradient_signal_diagnostic_smoke",
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
        log_signals=True,
        log_gradient_signals=True,
    )

    run = run_gradient_signal_diagnostic(config)
    loaded = load_experiment_artifacts(run.artifacts.run_dir)
    gradient_summary = loaded["gradient_signals"]["summary"]

    assert loaded["config"]["run"]["method_name"] == "gradient_signal_diagnostic"
    assert loaded["config"]["run"]["method"]["diagnostic_only"]
    assert loaded["config"]["run"]["signals"]["gradient_signals_enabled"]
    assert gradient_summary["row_count"] > 0
    assert gradient_summary["counts_by_observation_type"] == {"seen_task_eval": 48}
    assert (
        loaded["metrics"]["method_metrics"]["gradient_signals"]["row_count"]
        == gradient_summary["row_count"]
    )


def test_gradient_signal_diagnostic_config_loader_reads_smoke_config():
    config = load_config("configs/experiments/gradient_signal_diagnostic_smoke.yaml")

    assert config.method_name == "gradient_signal_diagnostic"
    assert config.smoke is True
    assert config.replay_capacity == 12
    assert config.replay_batch_size == 4
    assert config.log_gradient_signals is True
    assert config.hidden_dims == (32,)
