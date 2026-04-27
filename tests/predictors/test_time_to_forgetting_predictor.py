from src.predictors import (
    PRIMARY_TIME_TARGET,
    build_time_feature_rows,
    build_time_to_forgetting_report,
    evaluate_time_to_forgetting_heuristics,
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


def _time_row(
    *,
    sample_id,
    source_task_id,
    anchor_trained_task_id,
    anchor_correct,
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
        "anchor_correct": anchor_correct,
        "anchor_loss": 0.5,
        "anchor_confidence": 0.5,
        "anchor_target_probability": 0.5,
        "anchor_uncertainty": 0.5,
        "eligible_for_time_to_forgetting": anchor_correct,
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
            time_rows.append(
                _time_row(
                    sample_id=sample_id,
                    source_task_id=source_task_id,
                    anchor_trained_task_id=task_id,
                    anchor_correct=True,
                    event_observed=True,
                    step_delta=10 if high_risk else 30,
                    task_delta=1 if high_risk else 3,
                )
            )

    return {"rows": signal_rows}, {"rows": time_rows}


def test_time_feature_builder_uses_only_pre_anchor_history():
    signal_payload, time_payload = _payloads()

    rows = build_time_feature_rows(
        signal_payload=signal_payload,
        time_payload=time_payload,
    )
    high_risk_row = next(row for row in rows if row["sample_id"] == 10)

    assert high_risk_row["leakage_safe"] is True
    assert high_risk_row["history_eval_count"] == 2
    assert high_risk_row["anchor_loss"] == 1.4
    assert high_risk_row["first_observed_forgetting_step_delta"] == 10
    assert high_risk_row["event_observed"] is True


def test_time_heuristic_report_uses_temporal_holdout():
    signal_payload, time_payload = _payloads()
    rows = build_time_feature_rows(signal_payload=signal_payload, time_payload=time_payload)

    report = evaluate_time_to_forgetting_heuristics(rows)

    assert report["target"] == PRIMARY_TIME_TARGET
    assert report["temporal_split"] == {
        "train_anchor_task_max": 1,
        "test_anchor_task_min": 2,
    }
    assert report["test"]["n"] == 8
    assert (
        report["estimators"]["risk_scaled_anchor_loss"][
            "mae_task_delta_on_observed_events"
        ]
        == 0.0
    )


def test_full_time_report_records_feature_summary():
    signal_payload, time_payload = _payloads()

    report = build_time_to_forgetting_report(
        signal_payload=signal_payload,
        time_payload=time_payload,
    )

    assert report["schema_version"] == 1
    assert report["feature_summary"]["row_count"] == 16
    assert report["feature_summary"]["eligible_row_count"] == 16
    assert report["feature_summary"]["event_observed_count"] == 16
    assert report["target"] == PRIMARY_TIME_TARGET
