from src.predictors import (
    PRIMARY_TARGET,
    build_feature_rows,
    build_predictor_report,
    evaluate_heuristics,
    evaluate_logistic_predictor,
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
    anchor_correct,
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
            signal_rows.append(
                _signal_row(
                    sample_id=sample_id,
                    source_task_id=source_task_id,
                    trained_task_id=task_id + 1,
                    correct=not high_risk,
                    loss=2.0 if high_risk else 0.3,
                    target_probability=0.05 if high_risk else 0.7,
                    confidence=0.7,
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

    return {"rows": signal_rows}, {"rows": label_rows}


def test_feature_builder_uses_only_pre_anchor_history():
    signal_payload, label_payload = _payloads()

    rows = build_feature_rows(
        signal_payload=signal_payload,
        label_payload=label_payload,
    )
    high_risk_row = next(row for row in rows if row["sample_id"] == 10)

    assert high_risk_row["leakage_safe"] is True
    assert high_risk_row["history_eval_count"] == 2
    assert high_risk_row["anchor_loss"] == 1.4
    assert high_risk_row["max_loss_so_far"] == 1.4
    assert high_risk_row["max_loss_so_far"] != 2.0
    assert high_risk_row["forgot_any_future"] is True


def test_heuristic_report_uses_temporal_holdout():
    signal_payload, label_payload = _payloads()
    rows = build_feature_rows(signal_payload=signal_payload, label_payload=label_payload)

    report = evaluate_heuristics(rows)

    assert report["target"] == PRIMARY_TARGET
    assert report["temporal_split"] == {
        "train_anchor_task_max": 1,
        "test_anchor_task_min": 2,
    }
    assert report["test"]["n"] == 8
    assert report["heuristics"]["anchor_loss"]["average_precision"] == 1.0
    assert report["heuristics"]["combined_signal"]["average_precision"] == 1.0


def test_logistic_predictor_fits_when_temporal_train_has_two_classes():
    signal_payload, label_payload = _payloads()
    rows = build_feature_rows(signal_payload=signal_payload, label_payload=label_payload)

    report = evaluate_logistic_predictor(rows)

    assert report["status"] == "fit"
    assert report["metrics"]["average_precision"] == 1.0


def test_full_predictor_report_records_sources_and_feature_summary():
    signal_payload, label_payload = _payloads()

    report = build_predictor_report(
        signal_payload=signal_payload,
        label_payload=label_payload,
    )

    assert report["schema_version"] == 1
    assert report["feature_summary"]["row_count"] == 16
    assert report["feature_summary"]["eligible_row_count"] == 16
    assert report["target"] == PRIMARY_TARGET
