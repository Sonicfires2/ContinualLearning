import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.signals import (
    PRIMARY_TIME_TARGET,
    TIME_TO_FORGETTING_SCHEMA_VERSION,
    build_time_to_forgetting_artifact,
    save_time_to_forgetting_artifact,
)


def _eval_row(
    *,
    sample_id,
    source_task_id,
    trained_task_id,
    correct,
    loss,
    target_probability,
    confidence,
    class_id=None,
):
    return {
        "sample_id": sample_id,
        "source_task_id": source_task_id,
        "original_class_id": class_id if class_id is not None else source_task_id,
        "within_task_label": 0,
        "original_index": sample_id,
        "split": "test",
        "target": class_id if class_id is not None else source_task_id,
        "observation_type": "seen_task_eval",
        "trained_task_id": trained_task_id,
        "evaluated_task_id": source_task_id,
        "epoch": None,
        "global_step": trained_task_id * 10,
        "is_replay": False,
        "replay_count": 0,
        "last_replay_step": None,
        "loss": loss,
        "predicted_class": class_id if correct else 99,
        "correct": correct,
        "confidence": confidence,
        "target_probability": target_probability,
        "uncertainty": 1.0 - confidence,
        "entropy": 0.5,
    }


def _signal_payload():
    return {
        "schema_version": 1,
        "summary": {"row_count": 9},
        "rows": [
            _eval_row(
                sample_id=1,
                source_task_id=0,
                trained_task_id=0,
                correct=True,
                loss=0.1,
                target_probability=0.9,
                confidence=0.9,
            ),
            _eval_row(
                sample_id=1,
                source_task_id=0,
                trained_task_id=1,
                correct=True,
                loss=0.2,
                target_probability=0.8,
                confidence=0.8,
            ),
            _eval_row(
                sample_id=1,
                source_task_id=0,
                trained_task_id=2,
                correct=False,
                loss=1.3,
                target_probability=0.1,
                confidence=0.7,
            ),
            _eval_row(
                sample_id=2,
                source_task_id=0,
                trained_task_id=0,
                correct=True,
                loss=0.2,
                target_probability=0.8,
                confidence=0.8,
                class_id=1,
            ),
            _eval_row(
                sample_id=2,
                source_task_id=0,
                trained_task_id=1,
                correct=True,
                loss=0.25,
                target_probability=0.75,
                confidence=0.75,
                class_id=1,
            ),
            _eval_row(
                sample_id=2,
                source_task_id=0,
                trained_task_id=2,
                correct=True,
                loss=0.3,
                target_probability=0.7,
                confidence=0.7,
                class_id=1,
            ),
            _eval_row(
                sample_id=3,
                source_task_id=1,
                trained_task_id=1,
                correct=False,
                loss=2.0,
                target_probability=0.1,
                confidence=0.6,
                class_id=2,
            ),
            _eval_row(
                sample_id=3,
                source_task_id=1,
                trained_task_id=2,
                correct=False,
                loss=2.5,
                target_probability=0.05,
                confidence=0.7,
                class_id=2,
            ),
            {"observation_type": "current_train", "sample_id": 999},
        ],
    }


def test_build_time_to_forgetting_targets_with_censoring():
    artifact = build_time_to_forgetting_artifact(_signal_payload())

    assert artifact["schema_version"] == TIME_TO_FORGETTING_SCHEMA_VERSION
    assert artifact["primary_time_target"] == PRIMARY_TIME_TARGET
    assert artifact["summary"]["anchor_count"] == 5
    assert artifact["summary"]["eligible_for_time_to_forgetting_count"] == 4
    assert artifact["summary"]["event_observed_count"] == 2
    assert artifact["summary"]["right_censored_count"] == 2
    assert artifact["summary"]["not_retained_at_anchor_count"] == 1

    first = artifact["rows"][0]
    assert first["sample_id"] == 1
    assert first["event_observed"] is True
    assert first["first_observed_forgetting_task_delta"] == 2
    assert first["first_observed_forgetting_step_delta"] == 20
    assert first["interval_lower_task_delta"] == 1
    assert first["interval_upper_task_delta"] == 2
    assert first["interval_censored"] is True
    assert first["right_censored"] is False

    right_censored = next(
        row
        for row in artifact["rows"]
        if row["sample_id"] == 2 and row["anchor_trained_task_id"] == 0
    )
    assert right_censored["event_observed"] is False
    assert right_censored["right_censored"] is True
    assert right_censored["observed_survival_task_delta"] == 2
    assert right_censored["interval_upper_task_delta"] is None

    not_retained = artifact["rows"][-1]
    assert not_retained["censoring_type"] == "not_retained_at_anchor"
    assert not_retained["eligible_for_time_to_forgetting"] is False


def test_time_builder_rejects_eval_rows_with_future_leakage_shape():
    payload = {
        "rows": [
            _eval_row(
                sample_id=1,
                source_task_id=1,
                trained_task_id=0,
                correct=True,
                loss=0.1,
                target_probability=0.9,
                confidence=0.9,
            )
        ]
    }

    with pytest.raises(ValueError, match="before its source task"):
        build_time_to_forgetting_artifact(payload)


def test_save_time_to_forgetting_artifact_records_input_hash():
    workdir = Path(".tmp") / "test_time_to_forgetting" / uuid4().hex
    workdir.mkdir(parents=True, exist_ok=False)
    signal_path = workdir / "sample_signals.json"
    output_path = workdir / "time_to_forgetting_targets.json"
    signal_path.write_text(json.dumps(_signal_payload()), encoding="utf-8")

    artifact = save_time_to_forgetting_artifact(
        signal_path=signal_path,
        output_path=output_path,
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert artifact["source_signal_artifact"]["sha256"]
    assert saved["source_signal_artifact"]["path"] == str(signal_path)
    assert saved["summary"]["anchor_count"] == 5
