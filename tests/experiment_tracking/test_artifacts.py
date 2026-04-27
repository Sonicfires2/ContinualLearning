import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.experiment_tracking.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ExperimentRunConfig,
    load_experiment_artifacts,
    save_experiment_artifacts,
    summarize_training_result,
    validate_accuracy_matrix,
)
from src.training.continual import ContinualTrainingResult


def _run_config(run_name="smoke-run"):
    return ExperimentRunConfig(
        protocol_id="core_split_cifar100_v2",
        method_name="fine_tuning",
        seed=123,
        run_name=run_name,
        dataset={
            "name": "fixture_split_cifar100",
            "task_count": 3,
            "classes_per_task": 2,
            "split_seed": 0,
        },
        model={"name": "linear_fixture", "num_parameters": 6},
        trainer={"epochs_per_task": 1, "batch_size": 4, "eval_batch_size": 4},
        evaluation={"schedule": "evaluate_all_seen_tasks_after_each_task"},
        task_split={"class_order": [0, 1, 2, 3, 4, 5]},
        method={"description": "sequential fine-tuning without replay"},
    )


def _training_result():
    return ContinualTrainingResult(
        accuracy_matrix=[
            [0.90, None, None],
            [0.70, 0.80, None],
            [0.60, 0.75, 0.85],
        ],
        train_losses=[1.2, 0.8, 0.4],
        training_time_seconds=3.5,
        task_count=3,
    )


def _workspace_tmp_dir():
    path = Path(".tmp") / "test_artifacts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_summarize_training_result_matches_research_metrics():
    summary = summarize_training_result(_training_result())

    assert summary["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert summary["final_accuracy"] == pytest.approx((0.60 + 0.75 + 0.85) / 3)
    assert summary["average_forgetting"] == pytest.approx(
        ((0.90 - 0.60) + (0.80 - 0.75)) / 2
    )
    assert summary["forgetting_by_task"] == {
        "0": pytest.approx(0.30),
        "1": pytest.approx(0.05),
    }
    assert summary["train_loss_count"] == 3


def test_save_and_load_experiment_artifacts_round_trip():
    output_root = _workspace_tmp_dir()

    paths = save_experiment_artifacts(
        output_root=output_root,
        run_config=_run_config(),
        result=_training_result(),
        extra_metadata={"purpose": "unit-test"},
        extra_json_artifacts={
            "sample_signals": {
                "schema_version": 1,
                "rows": [{"sample_id": 1, "loss": 0.4}],
            }
        },
        environment={"python": {"version": "test"}, "torch": {"available": False}},
    )

    loaded = load_experiment_artifacts(paths.run_dir)

    assert paths.manifest.exists()
    assert paths.config.exists()
    assert paths.metrics.exists()
    assert paths.accuracy_matrix.exists()
    assert paths.train_losses.exists()
    assert paths.environment.exists()
    assert paths.extra["sample_signals"].exists()
    assert loaded["manifest"]["metadata"]["method_name"] == "fine_tuning"
    assert "sample_signals" in loaded["manifest"]["artifacts"]
    assert loaded["manifest"]["metadata"]["extra"] == {"purpose": "unit-test"}
    assert loaded["config"]["run"]["seed"] == 123
    assert loaded["metrics"]["final_accuracy"] == pytest.approx(
        (0.60 + 0.75 + 0.85) / 3
    )
    assert loaded["accuracy_matrix"]["accuracy_matrix"] == [
        [0.90, None, None],
        [0.70, 0.80, None],
        [0.60, 0.75, 0.85],
    ]
    assert loaded["train_losses"]["train_losses"] == [1.2, 0.8, 0.4]
    assert loaded["sample_signals"]["rows"] == [{"loss": 0.4, "sample_id": 1}]


def test_save_refuses_to_overwrite_existing_artifacts():
    output_root = _workspace_tmp_dir()

    save_experiment_artifacts(
        output_root=output_root,
        run_config=_run_config(run_name="same-run"),
        result=_training_result(),
        environment={"python": {"version": "test"}},
    )

    with pytest.raises(FileExistsError):
        save_experiment_artifacts(
            output_root=output_root,
            run_config=_run_config(run_name="same-run"),
            result=_training_result(),
            environment={"python": {"version": "test"}},
        )


def test_accuracy_matrix_validation_rejects_future_task_leakage():
    with pytest.raises(ValueError, match="future-task value"):
        validate_accuracy_matrix(
            [
                [0.9, 0.1],
                [0.8, 0.7],
            ]
        )


def test_accuracy_matrix_validation_rejects_missing_seen_task_value():
    with pytest.raises(ValueError, match="missing a seen-task value"):
        validate_accuracy_matrix(
            [
                [0.9, None],
                [None, 0.7],
            ]
        )


def test_artifact_hash_verification_catches_tampering():
    output_root = _workspace_tmp_dir()

    paths = save_experiment_artifacts(
        output_root=output_root,
        run_config=_run_config(),
        result=_training_result(),
        environment={"python": {"version": "test"}},
    )

    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
    metrics["final_accuracy"] = 0.0
    paths.metrics.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_experiment_artifacts(paths.run_dir)
