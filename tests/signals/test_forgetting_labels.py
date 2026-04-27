import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.signals import (
    FORGETTING_LABEL_SCHEMA_VERSION,
    PRIMARY_FORGETTING_LABEL,
    build_forgetting_label_artifact,
    save_forgetting_label_artifact,
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
                correct=False,
                loss=1.2,
                target_probability=0.2,
                confidence=0.7,
            ),
            _eval_row(
                sample_id=1,
                source_task_id=0,
                trained_task_id=2,
                correct=True,
                loss=0.3,
                target_probability=0.8,
                confidence=0.8,
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
            {
                "observation_type": "current_train",
                "sample_id": 999,
            },
        ],
    }


def test_build_forgetting_labels_from_future_eval_changes():
    artifact = build_forgetting_label_artifact(_signal_payload())

    assert artifact["schema_version"] == FORGETTING_LABEL_SCHEMA_VERSION
    assert artifact["primary_label"] == PRIMARY_FORGETTING_LABEL
    assert artifact["summary"]["anchor_count"] == 5
    assert artifact["summary"]["eligible_for_binary_forgetting_count"] == 3
    assert artifact["summary"]["forgot_any_future_count"] == 1
    assert artifact["summary"]["forgot_next_eval_count"] == 1
    assert artifact["summary"]["forgot_final_eval_count"] == 0

    first = artifact["rows"][0]
    assert first["sample_id"] == 1
    assert first["anchor_trained_task_id"] == 0
    assert first["future_eval_count"] == 2
    assert first["forgot_any_future"] is True
    assert first["forgot_next_eval"] is True
    assert first["forgot_final_eval"] is False
    assert first["max_future_loss_increase"] == pytest.approx(1.1)
    assert first["max_future_target_probability_drop"] == pytest.approx(0.7)
    assert first["leakage_safe"] is True

    never_learned_anchor = artifact["rows"][-1]
    assert never_learned_anchor["sample_id"] == 3
    assert never_learned_anchor["eligible_for_binary_forgetting"] is False
    assert never_learned_anchor["forgot_any_future"] is False


def test_label_builder_rejects_eval_rows_with_future_leakage_shape():
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
        build_forgetting_label_artifact(payload)


def test_save_forgetting_label_artifact_records_input_hash():
    workdir = Path(".tmp") / "test_forgetting_labels" / uuid4().hex
    workdir.mkdir(parents=True, exist_ok=False)
    signal_path = workdir / "sample_signals.json"
    output_path = workdir / "forgetting_labels.json"
    signal_path.write_text(json.dumps(_signal_payload()), encoding="utf-8")

    artifact = save_forgetting_label_artifact(
        signal_path=signal_path,
        output_path=output_path,
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert artifact["source_signal_artifact"]["sha256"]
    assert saved["source_signal_artifact"]["path"] == str(signal_path)
    assert saved["summary"]["anchor_count"] == 5
