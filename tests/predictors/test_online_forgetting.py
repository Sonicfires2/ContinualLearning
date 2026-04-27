from src.predictors import train_online_forgetting_risk_scorer


def _signal_row(
    *,
    sample_id,
    trained_task_id,
    high_risk,
):
    return {
        "sample_id": sample_id,
        "source_task_id": 0,
        "original_class_id": sample_id,
        "within_task_label": 0,
        "original_index": sample_id,
        "split": "test",
        "target": sample_id,
        "observation_type": "seen_task_eval",
        "trained_task_id": trained_task_id,
        "evaluated_task_id": 0,
        "epoch": None,
        "global_step": trained_task_id * 10,
        "is_replay": False,
        "replay_count": 0,
        "last_replay_step": None,
        "loss": 1.4 if high_risk else 0.2,
        "predicted_class": sample_id,
        "correct": True,
        "confidence": 0.55 if high_risk else 0.85,
        "target_probability": 0.15 if high_risk else 0.8,
        "uncertainty": 0.45 if high_risk else 0.15,
        "entropy": 0.5,
    }


def _label_row(*, sample_id, anchor_trained_task_id, high_risk):
    return {
        "sample_id": sample_id,
        "split": "test",
        "source_task_id": 0,
        "original_class_id": sample_id,
        "within_task_label": 0,
        "original_index": sample_id,
        "target": sample_id,
        "anchor_trained_task_id": anchor_trained_task_id,
        "anchor_evaluated_task_id": 0,
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
        "forgot_next_eval": high_risk,
        "forgot_final_eval": high_risk,
        "forgot_any_future": high_risk,
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
                        trained_task_id=task_id - 1,
                        high_risk=high_risk,
                    )
                )
            signal_rows.append(
                _signal_row(
                    sample_id=sample_id,
                    trained_task_id=task_id,
                    high_risk=high_risk,
                )
            )
            label_rows.append(
                _label_row(
                    sample_id=sample_id,
                    anchor_trained_task_id=task_id,
                    high_risk=high_risk,
                )
            )
    return {"rows": signal_rows}, {"rows": label_rows}


def test_online_forgetting_risk_scorer_scores_high_risk_above_low_risk():
    signal_payload, label_payload = _payloads()

    scorer = train_online_forgetting_risk_scorer(
        signal_payload=signal_payload,
        label_payload=label_payload,
    )
    high_score = scorer.score(
        {
            "anchor_loss": 1.4,
            "anchor_uncertainty": 0.45,
            "anchor_confidence": 0.55,
            "anchor_target_probability": 0.15,
            "history_eval_count": 2,
            "has_previous_eval": 1,
            "previous_correct": 1,
            "previous_loss": 0.6,
            "previous_uncertainty": 0.3,
            "previous_target_probability": 0.4,
            "loss_delta_from_previous": 0.8,
            "uncertainty_delta_from_previous": 0.15,
            "target_probability_delta_from_previous": -0.25,
            "loss_increase_from_previous": 0.8,
            "target_probability_drop_from_previous": 0.25,
            "max_loss_so_far": 1.4,
            "min_target_probability_so_far": 0.15,
            "max_uncertainty_so_far": 0.45,
            "correct_history_rate": 1.0,
            "tasks_since_source": 2,
            "anchor_task_progress": 0.5,
        }
    )
    low_score = scorer.score(
        {
            "anchor_loss": 0.2,
            "anchor_uncertainty": 0.15,
            "anchor_confidence": 0.85,
            "anchor_target_probability": 0.8,
            "history_eval_count": 2,
            "has_previous_eval": 1,
            "previous_correct": 1,
            "previous_loss": 0.2,
            "previous_uncertainty": 0.15,
            "previous_target_probability": 0.8,
            "loss_delta_from_previous": 0.0,
            "uncertainty_delta_from_previous": 0.0,
            "target_probability_delta_from_previous": 0.0,
            "loss_increase_from_previous": 0.0,
            "target_probability_drop_from_previous": 0.0,
            "max_loss_so_far": 0.2,
            "min_target_probability_so_far": 0.8,
            "max_uncertainty_so_far": 0.15,
            "correct_history_rate": 1.0,
            "tasks_since_source": 2,
            "anchor_task_progress": 0.5,
        }
    )

    assert high_score > low_score
    assert scorer.to_json_metadata()["model_family"] == "logistic_regression"
