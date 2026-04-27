import json
from pathlib import Path
from uuid import uuid4

from src.predictors import (
    LEARNED_PREDICTOR_REPORT_SCHEMA_VERSION,
    build_feature_rows,
    build_learned_predictor_report,
    evaluate_binary_learned_models,
    evaluate_continuous_forgetting_models,
    evaluate_time_to_forgetting_learned_models,
    save_learned_predictor_report,
)
from src.predictors.time_to_forgetting import build_time_feature_rows


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
    anchor_correct,
    forgot_any_future,
):
    high_value = 1.5 if forgot_any_future else 0.05
    low_value = 0.15 if forgot_any_future else 0.02
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
        "anchor_correct": anchor_correct,
        "anchor_loss": 0.5,
        "anchor_confidence": 0.5,
        "anchor_target_probability": 0.5,
        "anchor_uncertainty": 0.5,
        "eligible_for_binary_forgetting": anchor_correct,
        "future_eval_count": 1,
        "next_trained_task_id": anchor_trained_task_id + 1,
        "final_trained_task_id": anchor_trained_task_id + 1,
        "forgot_next_eval": forgot_any_future,
        "forgot_final_eval": forgot_any_future,
        "forgot_any_future": forgot_any_future,
        "next_loss_delta": high_value,
        "final_loss_delta": high_value,
        "max_future_loss_increase": high_value,
        "next_target_probability_drop": high_value,
        "final_target_probability_drop": high_value,
        "max_future_target_probability_drop": high_value,
        "next_confidence_drop": low_value,
        "final_confidence_drop": low_value,
        "max_future_confidence_drop": low_value,
        "future_min_target_probability": 0.1 if forgot_any_future else 0.7,
        "future_max_loss": 2.0 if forgot_any_future else 0.3,
        "label_uses_future_after_task_id": anchor_trained_task_id,
        "leakage_safe": True,
    }


def _time_row(
    *,
    sample_id,
    source_task_id,
    anchor_trained_task_id,
    event_observed,
    step_delta,
    task_delta,
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
        "eligible_for_time_to_forgetting": True,
        "future_eval_count": 1,
        "event_observed": event_observed,
        "censoring_type": (
            "event_observed_interval_censored"
            if event_observed
            else "right_censored"
        ),
        "first_forgetting_trained_task_id": (
            anchor_trained_task_id + task_delta if event_observed else None
        ),
        "first_forgetting_global_step": (
            anchor_trained_task_id * 10 + step_delta if event_observed else None
        ),
        "first_observed_forgetting_task_delta": task_delta if event_observed else None,
        "first_observed_forgetting_step_delta": step_delta if event_observed else None,
        "last_observed_correct_trained_task_id": anchor_trained_task_id,
        "last_observed_correct_global_step": anchor_trained_task_id * 10,
        "interval_lower_task_delta": 0 if event_observed else task_delta,
        "interval_lower_step_delta": 0 if event_observed else step_delta,
        "interval_upper_task_delta": task_delta if event_observed else None,
        "interval_upper_step_delta": step_delta if event_observed else None,
        "survived_until_trained_task_id": anchor_trained_task_id,
        "survived_until_global_step": anchor_trained_task_id * 10,
        "observed_survival_task_delta": 0 if event_observed else task_delta,
        "observed_survival_step_delta": 0 if event_observed else step_delta,
        "interval_censored": event_observed,
        "right_censored": not event_observed,
        "leakage_safe": True,
    }


def _payloads():
    signal_rows = []
    label_rows = []
    time_rows = []
    for task_id in range(4):
        for offset in range(4):
            sample_id = task_id * 10 + offset
            high_risk = offset in {0, 1}
            source_task_id = 0
            previous_loss = 0.2 if not high_risk else 0.6
            anchor_loss = 0.25 if not high_risk else 1.4
            if task_id > 0:
                signal_rows.append(
                    _signal_row(
                        sample_id=sample_id,
                        source_task_id=source_task_id,
                        trained_task_id=task_id - 1,
                        correct=True,
                        loss=previous_loss,
                        target_probability=0.8 if not high_risk else 0.4,
                        confidence=0.8 if not high_risk else 0.7,
                    )
                )
            signal_rows.append(
                _signal_row(
                    sample_id=sample_id,
                    source_task_id=source_task_id,
                    trained_task_id=task_id,
                    correct=True,
                    loss=anchor_loss,
                    target_probability=0.75 if not high_risk else 0.15,
                    confidence=0.75 if not high_risk else 0.55,
                )
            )
            label_rows.append(
                _label_row(
                    sample_id=sample_id,
                    source_task_id=source_task_id,
                    anchor_trained_task_id=task_id,
                    anchor_correct=True,
                    forgot_any_future=high_risk,
                )
            )
            time_rows.append(
                _time_row(
                    sample_id=sample_id,
                    source_task_id=source_task_id,
                    anchor_trained_task_id=task_id,
                    event_observed=True,
                    step_delta=10 if high_risk else 30,
                    task_delta=1 if high_risk else 3,
                )
            )

    return {"rows": signal_rows}, {"rows": label_rows}, {"rows": time_rows}


def test_binary_learned_models_fit_logistic_and_svm_on_temporal_split():
    signal_payload, label_payload, _ = _payloads()
    rows = build_feature_rows(signal_payload=signal_payload, label_payload=label_payload)

    report = evaluate_binary_learned_models(rows)

    assert report["models"]["logistic_regression"]["status"] == "fit"
    assert report["models"]["linear_svm_classifier"]["status"] == "fit"
    assert (
        report["models"]["logistic_regression"]["metrics"]["average_precision"] == 1.0
    )
    assert report["models"]["logistic_regression"]["threshold_behavior"]


def test_continuous_models_fit_label_deterioration_targets():
    signal_payload, label_payload, _ = _payloads()
    rows = build_feature_rows(signal_payload=signal_payload, label_payload=label_payload)

    report = evaluate_continuous_forgetting_models(rows)
    target = report["targets"]["max_future_loss_increase"]

    assert target["status"] == "fit"
    assert target["models"]["linear_regression"]["status"] == "fit"
    assert target["models"]["ridge_regression"]["metrics"]["mae"] is not None
    assert target["models"]["linear_svm_regressor"]["metrics"]["rmse"] is not None


def test_time_to_forgetting_learned_models_fit_observed_event_targets():
    signal_payload, _, time_payload = _payloads()
    rows = build_time_feature_rows(
        signal_payload=signal_payload,
        time_payload=time_payload,
    )

    report = evaluate_time_to_forgetting_learned_models(rows)

    assert report["status"] == "fit"
    assert report["models"]["ridge_regression"]["metrics"]["mae"] is not None
    assert report["target_scope"].startswith("observed events only")


def test_full_learned_predictor_report_compares_against_heuristics():
    signal_payload, label_payload, time_payload = _payloads()

    report = build_learned_predictor_report(
        signal_payload=signal_payload,
        label_payload=label_payload,
        time_payload=time_payload,
    )

    assert report["schema_version"] == LEARNED_PREDICTOR_REPORT_SCHEMA_VERSION
    assert report["feature_summary"]["eligible_binary_row_count"] == 16
    assert (
        report["comparison_summary"]["best_binary_model_by_average_precision"][
            "average_precision"
        ]
        == 1.0
    )
    assert report["time_to_forgetting_report"]["status"] == "fit"


def test_save_learned_predictor_report_writes_json():
    signal_payload, label_payload, time_payload = _payloads()
    tmp_path = Path(".tmp") / "test_learned_forgetting" / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    signal_path = tmp_path / "sample_signals.json"
    label_path = tmp_path / "forgetting_labels.json"
    time_path = tmp_path / "time_to_forgetting_targets.json"
    output_path = tmp_path / "learned_forgetting_predictor_report.json"
    for path, payload in (
        (signal_path, signal_payload),
        (label_path, label_payload),
        (time_path, time_payload),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = save_learned_predictor_report(
        signal_path=signal_path,
        label_path=label_path,
        time_path=time_path,
        output_path=output_path,
    )

    assert Path(output_path).exists()
    assert report["source_signal_artifact"]["path"] == str(signal_path)
