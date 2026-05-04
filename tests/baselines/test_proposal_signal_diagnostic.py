from src.baselines.proposal_signal_diagnostic import (
    ProposalSignalDiagnosticConfig,
    run_proposal_signal_diagnostic,
)


def test_proposal_signal_diagnostic_smoke_logs_all_signal_families():
    run = run_proposal_signal_diagnostic(
        ProposalSignalDiagnosticConfig(
            run_name="proposal_signal_diagnostic_test_smoke",
            output_root=".tmp/test_baseline_runs",
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
            log_gradient_signals=True,
            log_representation_signals=True,
        )
    )

    assert run.artifacts.run_dir.exists()
    assert "signals" in run.result.method_metrics
    assert "gradient_signals" in run.result.method_metrics
    assert "representation_signals" in run.result.method_metrics
    assert run.result.method_metrics["representation_signals"]["row_count"] > 0
    assert (run.artifacts.run_dir / "representation_signals.json").exists()
