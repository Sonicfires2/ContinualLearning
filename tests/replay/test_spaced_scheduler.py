import torch

from src.replay import ReplayItem, SpacedReplayScheduler, SpacedReplaySchedulerConfig


def _item(sample_id, task_id=0, added_at_step=0):
    return ReplayItem(
        x=torch.tensor([float(sample_id)]),
        target=sample_id,
        sample_id=sample_id,
        task_id=task_id,
        original_class_id=sample_id,
        within_task_label=0,
        original_index=sample_id,
        split="train",
        added_at_task=task_id,
        added_at_step=added_at_step,
    )


def _batch(sample_ids, task_id=0):
    return {
        "sample_id": torch.tensor(sample_ids, dtype=torch.long),
        "task_id": torch.tensor([task_id for _ in sample_ids], dtype=torch.long),
        "original_class_id": torch.tensor(sample_ids, dtype=torch.long),
        "within_task_label": torch.zeros(len(sample_ids), dtype=torch.long),
        "original_index": torch.tensor(sample_ids, dtype=torch.long),
    }


def test_scheduler_maps_higher_risk_to_shorter_interval():
    scheduler = SpacedReplayScheduler(
        SpacedReplaySchedulerConfig(min_interval_steps=1, max_interval_steps=10)
    )
    logits = torch.tensor(
        [
            [5.0, -2.0],
            [-1.0, 1.0],
        ]
    )
    targets = torch.tensor([0, 0])

    scheduler.observe_batch(
        logits=logits,
        targets=targets,
        batch=_batch([1, 2]),
        global_step=0,
        is_replay=False,
    )

    low_risk = scheduler.state_for(1)
    high_risk = scheduler.state_for(2)
    assert low_risk is not None
    assert high_risk is not None
    assert high_risk.risk_score > low_risk.risk_score
    assert (
        high_risk.estimated_forgetting_time_steps
        < low_risk.estimated_forgetting_time_steps
    )


def test_scheduler_selects_due_items_before_budget_fill():
    scheduler = SpacedReplayScheduler(
        SpacedReplaySchedulerConfig(
            min_interval_steps=1,
            max_interval_steps=5,
            budget_mode="match_random_replay",
        )
    )
    scheduler.observe_batch(
        logits=torch.tensor([[0.0, 2.0], [4.0, -1.0]]),
        targets=torch.tensor([0, 0]),
        batch=_batch([1, 2]),
        global_step=0,
        is_replay=False,
    )
    selections = scheduler.select(
        items=[_item(1), _item(2)],
        global_step=4,
        batch_size=2,
    )

    assert len(selections) == 2
    assert selections[0].selection_reason == "due"
    assert selections[0].sample_id == 1
    assert selections[1].selection_reason in {"due", "budget_fill_near_due"}
    assert scheduler.to_json_payload()["summary"]["trace_row_count"] == 2


def test_scheduler_risk_gated_mode_skips_until_risk_and_due_gate_passes():
    scheduler = SpacedReplayScheduler(
        SpacedReplaySchedulerConfig(
            min_interval_steps=1,
            max_interval_steps=5,
            risk_threshold=0.5,
            budget_mode="risk_and_due",
        )
    )
    scheduler.observe_batch(
        logits=torch.tensor([[5.0, -1.0], [-5.0, 5.0]]),
        targets=torch.tensor([0, 0]),
        batch=_batch([1, 2]),
        global_step=0,
        is_replay=False,
    )

    early = scheduler.select(
        items=[_item(1), _item(2)],
        global_step=0,
        batch_size=2,
    )
    selected = scheduler.select(
        items=[_item(1), _item(2)],
        global_step=5,
        batch_size=2,
    )
    payload = scheduler.to_json_payload()

    assert early == []
    assert payload["summary"]["skipped_selection_event_count"] == 1
    assert payload["skipped_rows"][0]["skip_reason"] == "no_due_items"
    assert len(selected) == 1
    assert selected[0].sample_id == 2
    assert selected[0].selection_reason == "risk_threshold_and_due"


def test_scheduler_risk_ranked_mode_fills_budget_by_risk_without_threshold_skip():
    scheduler = SpacedReplayScheduler(
        SpacedReplaySchedulerConfig(
            min_interval_steps=1,
            max_interval_steps=5,
            risk_threshold=0.99,
            budget_mode="risk_ranked",
        )
    )
    scheduler.observe_batch(
        logits=torch.tensor([[5.0, -1.0], [-5.0, 5.0], [0.1, 0.2]]),
        targets=torch.tensor([0, 0, 0]),
        batch=_batch([1, 2, 3]),
        global_step=0,
        is_replay=False,
    )

    selected = scheduler.select(
        items=[_item(1), _item(2), _item(3)],
        global_step=0,
        batch_size=2,
    )
    payload = scheduler.to_json_payload()

    assert [selection.sample_id for selection in selected] == [2, 3]
    assert all(selection.selection_reason == "risk_ranked" for selection in selected)
    assert payload["summary"]["skipped_selection_event_count"] == 0
    assert payload["summary"]["trace_row_count"] == 2
