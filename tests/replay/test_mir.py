import torch
from torch import nn

from src.replay import ReplayItem, select_mir_replay_items


def _item(sample_id, target, x):
    return ReplayItem(
        x=torch.tensor(x, dtype=torch.float32),
        target=target,
        sample_id=sample_id,
        task_id=0,
        original_class_id=target,
        within_task_label=target,
        original_index=sample_id,
        split="train",
        added_at_task=0,
        added_at_step=0,
    )


def test_mir_selection_restores_model_after_virtual_update():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    candidates = [
        _item(1, 0, [1.0, 0.0]),
        _item(2, 1, [0.0, 1.0]),
        _item(3, 0, [0.5, 0.5]),
    ]

    selections = select_mir_replay_items(
        model=model,
        current_x=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        current_y=torch.tensor([0], dtype=torch.long),
        candidate_items=candidates,
        replay_batch_size=2,
        virtual_lr=0.1,
        global_step=5,
        device=torch.device("cpu"),
    )

    assert len(selections) == 2
    assert selections[0].interference_score >= selections[1].interference_score
    assert selections[0].candidate_rank == 1
    assert selections[0].candidate_count == 3
    for parameter, original in zip(model.parameters(), before, strict=True):
        assert torch.allclose(parameter, original)


def test_mir_selection_returns_empty_for_empty_candidates():
    model = nn.Linear(2, 2)

    selections = select_mir_replay_items(
        model=model,
        current_x=torch.zeros(1, 2),
        current_y=torch.zeros(1, dtype=torch.long),
        candidate_items=[],
        replay_batch_size=2,
        virtual_lr=0.1,
        global_step=0,
        device=torch.device("cpu"),
    )

    assert selections == []
