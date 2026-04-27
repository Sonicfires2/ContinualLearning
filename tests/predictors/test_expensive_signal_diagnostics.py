from pathlib import Path
from uuid import uuid4
import json

from src.predictors import (
    EXPENSIVE_SIGNAL_DIAGNOSTIC_SCHEMA_VERSION,
    augment_feature_rows_with_gradient_signals,
    build_expensive_signal_diagnostic_report,
    build_feature_rows,
    save_expensive_signal_diagnostic_report,
)


def _signal_row(
    *,
    sample_id,
    source_task_id,
    trained_task_id,
    correct,
    loss,
    target_probability,
    confidence,
):
    return {
        "sample_id": sample_id,
        "source_task_id": source_task_id,
        "original_class_id": sample_id,
        "within_task_label": 0,
        "original_index": sample_id,
        "split": "test",
        "target": sample_id,
        "observation_type": "seen_task_eval",
        "trained_task_id": trained_task_id,
        "evaluated_task_id": source_task_id,
        "epoch": None,
        "global_step": trained_task_id * 10,
        "is_replay": False,
        "replay_count": 0,
        "last_replay_step": None,
        "loss": loss,
        "predicted_class": sample_id if correct else 99,
        "correct": correct,
        "confidence": confidence,
        "target_probability": target_probability,
        "uncertainty": 1.0 - confidence,
        "entropy": 0.5,
    }


def _gradient_row(*, sample_id, source_task_id, trained_task_id, high):
    gradient = 2.0 if high else 0.2
    return {
        "sample_id": sample_id,
        "source_task_id": source_task_id,
        "original_class_id": sample_id,
        "within_task_label": 0,
        "original_index": sample_id,
        "split": "test",
        "target": sample_id,
        "observation_type": "seen_task_eval",
        "trained_task_id": trained_task_id,
        "evaluated_task_id": source_task_id,
        "epoch": None,
        "global_step": trained_task_id * 10,
        "is_replay": False,
        "loss": gradient,
        "predicted_class": sample_id,
        "correct": True,
        "target_probability": 0.2 if high else 0.8,
        "logit_gradient_l2": gradient / 2.0,
        "penultimate_activation_l2": gradient + 1.0,
        "classifier_weight_gradient_l2": gradient,
        "classifier_bias_gradient_l2": gradient / 2.0,
        "last_layer_gradient_l2": gradient,
    }


def _label_row(
    *,
    sample_id,
    source_task_id,
    anchor_trained_task_id,
    forgot_any_future,
):
    return {
        "sample_id": sample_id,
        "split": "test",
        "source_task_id": source_task_id,
        "original_class_id": sample_id,
        "within_task_label": 0,
        "original_index": sample_id,
        "target": sample_id,
        "anchor_trained_task_id": anchor_trained_task_id,
        "anchor_evaluated_task_id": source_task_id,
        "anchor_global_step": anchor_trained_task_id * 10,
        "anchor_correct": True,
        "anchor_loss": 0.5,
        "anchor_confidence": 0.5,
        "anchor_target_probability": 0.5,
        "anchor_uncertainty": 0.5,
        "eligible_for_binary_forgetting": True,
        "future_eval_count": 1,
        "next_trained_task_id": anchor_trained_task_id + 1,
        "final_trained_task_id": anchor_trained_task_id + 1,
        "forgot_next_eval": forgot_any_future,
        "forgot_final_eval": forgot_any_future,
        "forgot_any_future": forgot_any_future,
        "next_loss_delta": 1.5 if forgot_any_future else 0.05,
        "final_loss_delta": 1.5 if forgot_any_future else 0.05,
        "max_future_loss_increase": 1.5 if forgot_any_future else 0.05,
        "next_target_probability_drop": 1.5 if forgot_any_future else 0.05,
        "final_target_probability_drop": 1.5 if forgot_any_future else 0.05,
        "max_future_target_probability_drop": 1.5 if forgot_any_future else 0.05,
        "next_confidence_drop": 0.15 if forgot_any_future else 0.02,
        "final_confidence_drop": 0.15 if forgot_any_future else 0.02,
        "max_future_confidence_drop": 0.15 if forgot_any_future else 0.02,
        "future_min_target_probability": 0.1 if forgot_any_future else 0.7,
        "future_max_loss": 2.0 if forgot_any_future else 0.3,
        "label_uses_future_after_task_id": anchor_trained_task_id,
        "leakage_safe": True,
    }


def _payloads():
    signal_rows = []
    gradient_rows = []
    label_rows = []
    for task_id in range(4):
        for offset in range(4):
            sample_id = task_id * 10 + offset
            high_risk = offset in {0, 1}
            if task_id > 0:
                signal_rows.append(
                    _signal_row(
                        sample_id=sample_id,
                        source_task_id=0,
                        trained_task_id=task_id - 1,
                        correct=True,
                        loss=0.4,
                        target_probability=0.6,
                        confidence=0.7,
                    )
                )
                gradient_rows.append(
                    _gradient_row(
                        sample_id=sample_id,
                        source_task_id=0,
                        trained_task_id=task_id - 1,
                        high=high_risk,
                    )
                )
            signal_rows.append(
                _signal_row(
                    sample_id=sample_id,
                    source_task_id=0,
                    trained_task_id=task_id,
                    correct=True,
                    loss=0.5,
                    target_probability=0.5,
                    confidence=0.7,
                )
            )
            gradient_rows.append(
                _gradient_row(
                    sample_id=sample_id,
                    source_task_id=0,
                    trained_task_id=task_id,
                    high=high_risk,
                )
            )
            label_rows.append(
                _label_row(
                    sample_id=sample_id,
                    source_task_id=0,
                    anchor_trained_task_id=task_id,
                    forgot_any_future=high_risk,
                )
            )
    return {"rows": signal_rows}, {"rows": label_rows}, {"rows": gradient_rows}


def test_augment_feature_rows_with_gradient_signals_adds_anchor_fields():
    signal_payload, label_payload, gradient_payload = _payloads()
    rows = build_feature_rows(signal_payload=signal_payload, label_payload=label_payload)

    augmented = augment_feature_rows_with_gradient_signals(
        feature_rows=rows,
        gradient_payload=gradient_payload,
    )

    assert len(augmented) == len(rows)
    assert "anchor_last_layer_gradient_l2" in augmented[0]
    assert "last_layer_gradient_increase_from_previous" in augmented[0]


def test_expensive_signal_diagnostic_report_compares_gradient_group():
    signal_payload, label_payload, gradient_payload = _payloads()

    report = build_expensive_signal_diagnostic_report(
        signal_payload=signal_payload,
        label_payload=label_payload,
        gradient_payload=gradient_payload,
    )

    assert report["schema_version"] == EXPENSIVE_SIGNAL_DIAGNOSTIC_SCHEMA_VERSION
    assert report["feature_summary"]["gradient_augmented_row_count"] == 16
    assert "gradient_only" in report["ablation_report"]["feature_group_reports"]
    assert "build_replay_intervention_next" in report["recommendation"]


def test_save_expensive_signal_diagnostic_report_writes_json():
    signal_payload, label_payload, gradient_payload = _payloads()
    tmp_path = Path(".tmp") / "test_expensive_signal_diagnostics" / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    signal_path = tmp_path / "sample_signals.json"
    label_path = tmp_path / "forgetting_labels.json"
    gradient_path = tmp_path / "gradient_signals.json"
    output_path = tmp_path / "expensive_signal_diagnostic_report.json"
    signal_path.write_text(json.dumps(signal_payload), encoding="utf-8")
    label_path.write_text(json.dumps(label_payload), encoding="utf-8")
    gradient_path.write_text(json.dumps(gradient_payload), encoding="utf-8")

    report = save_expensive_signal_diagnostic_report(
        signal_path=signal_path,
        label_path=label_path,
        gradient_path=gradient_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert report["source_gradient_artifact"]["path"] == str(gradient_path)
