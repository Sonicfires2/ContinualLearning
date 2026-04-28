from src.predictors import MIRInterferenceDiagnosticLogger
from src.replay import MIRCandidateScore


def _candidate(sample_id: int, rank: int, score: float) -> MIRCandidateScore:
    return MIRCandidateScore(
        global_step=10,
        optimizer_step=11,
        sample_id=sample_id,
        source_task_id=0,
        original_class_id=sample_id,
        replay_count_before_selection=0,
        last_replay_step_before_selection=None,
        pre_update_loss=1.0,
        post_update_loss=1.0 + score,
        interference_score=score,
        candidate_rank=rank,
        candidate_count=4,
    )


def test_mir_interference_logger_measures_learned_risk_alignment():
    logger = MIRInterferenceDiagnosticLogger()
    candidates = [
        _candidate(1, 1, 0.5),
        _candidate(2, 2, 0.4),
        _candidate(3, 3, 0.1),
        _candidate(4, 4, 0.0),
    ]

    logger.record(
        candidate_scores=candidates,
        learned_risk_scores={1: 0.2, 2: 0.9, 3: 0.8, 4: 0.1},
        replay_batch_size=2,
    )

    summary = logger.summary()
    metrics = summary["learned_risk_predicts_mir_topk"]
    overlap = summary["event_topk_overlap"]

    assert summary["candidate_row_count"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["average_precision"] is not None
    assert overlap["mean_topk_overlap"] == 0.5
    assert overlap["mean_random_expected_topk_overlap"] == 0.5
    assert logger.to_json_payload()["rows"][0]["is_mir_topk"] is True

