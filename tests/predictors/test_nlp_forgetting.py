from src.predictors.nlp_forgetting import (
    NLP_FORGETTING_TARGET,
    build_nlp_forgetting_feature_rows,
    build_nlp_forgetting_report,
    evaluate_nlp_heuristics,
    evaluate_nlp_logistic_predictor,
)


def _row(*, sample_id, source_task, trained_task, correct, loss, target_probability):
    confidence = max(target_probability, 0.6)
    return {
        "sample_id": sample_id,
        "task_id": source_task,
        "eval_task_id": source_task,
        "trained_task_id": trained_task,
        "global_step": trained_task * 10,
        "original_class_id": source_task,
        "label": source_task,
        "loss": loss,
        "confidence": confidence,
        "target_probability": target_probability,
        "predicted_label": source_task if correct else 99,
        "correct": correct,
    }


def _payload():
    rows = []
    for anchor_task in range(4):
        for offset in range(4):
            high_risk = offset in {0, 1}
            sample_id = anchor_task * 10 + offset
            previous_task = max(0, anchor_task - 1)
            rows.append(
                _row(
                    sample_id=sample_id,
                    source_task=0,
                    trained_task=previous_task,
                    correct=True,
                    loss=0.2 if not high_risk else 0.5,
                    target_probability=0.8 if not high_risk else 0.5,
                )
            )
            rows.append(
                _row(
                    sample_id=sample_id,
                    source_task=0,
                    trained_task=anchor_task,
                    correct=True,
                    loss=0.25 if not high_risk else 1.5,
                    target_probability=0.75 if not high_risk else 0.1,
                )
            )
            rows.append(
                _row(
                    sample_id=sample_id,
                    source_task=0,
                    trained_task=anchor_task + 1,
                    correct=not high_risk,
                    loss=0.3 if not high_risk else 3.0,
                    target_probability=0.7 if not high_risk else 0.02,
                )
            )
    return {"rows": rows}


def test_nlp_feature_rows_use_future_only_for_labels():
    rows = build_nlp_forgetting_feature_rows(_payload())
    high_risk_row = next(
        row
        for row in rows
        if row["sample_id"] == 10 and row["anchor_trained_task_id"] == 1
    )

    assert high_risk_row["leakage_safe"] is True
    assert high_risk_row[NLP_FORGETTING_TARGET] is True
    assert high_risk_row["anchor_loss"] == 1.5
    assert high_risk_row["max_loss_so_far"] == 1.5
    assert high_risk_row["max_loss_so_far"] != 3.0
    assert high_risk_row["future_eval_count"] == 1


def test_nlp_heuristics_use_temporal_holdout():
    feature_rows = build_nlp_forgetting_feature_rows(_payload())

    report = evaluate_nlp_heuristics(feature_rows)

    assert report["target"] == NLP_FORGETTING_TARGET
    assert report["temporal_split"] == {
        "train_anchor_task_max": 1,
        "test_anchor_task_min": 2,
    }
    assert report["test"]["n"] > 0
    assert report["heuristics"]["anchor_loss"]["average_precision"] == 1.0
    assert report["heuristics"]["combined_signal"]["average_precision"] == 1.0


def test_nlp_logistic_predictor_fits_with_two_train_classes():
    feature_rows = build_nlp_forgetting_feature_rows(_payload())

    report = evaluate_nlp_logistic_predictor(feature_rows)

    assert report["status"] == "fit"
    assert report["metrics"]["average_precision"] == 1.0


def test_nlp_predictor_report_has_feature_summary():
    report = build_nlp_forgetting_report(_payload())

    assert report["schema_version"] == 1
    assert report["feature_summary"]["row_count"] > 0
    assert report["feature_summary"]["eligible_row_count"] > 0
    assert report["target"] == NLP_FORGETTING_TARGET

