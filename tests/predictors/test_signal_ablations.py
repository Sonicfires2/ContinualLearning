import json
from pathlib import Path
from uuid import uuid4

from src.predictors import (
    SIGNAL_ABLATION_REPORT_SCHEMA_VERSION,
    build_signal_ablation_report,
    build_feature_rows,
    evaluate_signal_ablations,
    save_signal_ablation_report,
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
                        loss=0.6 if high_risk else 0.2,
                        target_probability=0.4 if high_risk else 0.8,
                        confidence=0.7 if high_risk else 0.8,
                    )
                )
            signal_rows.append(
                _signal_row(
                    sample_id=sample_id,
                    source_task_id=0,
                    trained_task_id=task_id,
                    correct=True,
                    loss=1.4 if high_risk else 0.25,
                    target_probability=0.15 if high_risk else 0.75,
                    confidence=0.55 if high_risk else 0.75,
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
    return {"rows": signal_rows}, {"rows": label_rows}


def test_signal_ablation_report_ranks_feature_groups_and_thresholds():
    signal_payload, label_payload = _payloads()
    rows = build_feature_rows(signal_payload=signal_payload, label_payload=label_payload)

    report = evaluate_signal_ablations(rows)

    assert report["best_heuristic"]["average_precision"] == 1.0
    assert report["ranked_feature_groups"][0]["average_precision"] == 1.0
    assert (
        report["feature_group_reports"]["loss_only"]["model_report"]["models"][
            "logistic_regression"
        ]["status"]
        == "fit"
    )
    assert (
        report["all_features_logistic_threshold_recommendation"]["threshold"]
        is not None
    )


def test_full_signal_ablation_report_records_schema_and_feature_groups():
    signal_payload, label_payload = _payloads()

    report = build_signal_ablation_report(
        signal_payload=signal_payload,
        label_payload=label_payload,
    )

    assert report["schema_version"] == SIGNAL_ABLATION_REPORT_SCHEMA_VERSION
    assert report["feature_summary"]["eligible_binary_row_count"] == 16
    assert "history_delta" in report["feature_summary"]["feature_groups"]


def test_save_signal_ablation_report_writes_json():
    signal_payload, label_payload = _payloads()
    tmp_path = Path(".tmp") / "test_signal_ablations" / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    signal_path = tmp_path / "sample_signals.json"
    label_path = tmp_path / "forgetting_labels.json"
    output_path = tmp_path / "signal_ablation_report.json"
    signal_path.write_text(json.dumps(signal_payload), encoding="utf-8")
    label_path.write_text(json.dumps(label_payload), encoding="utf-8")

    report = save_signal_ablation_report(
        signal_path=signal_path,
        label_path=label_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert report["source_label_artifact"]["path"] == str(label_path)
