import torch
from torch import nn

from src.data.split_cifar100 import SplitCIFAR100TaskStream
from src.training.continual import ContinualTrainerConfig, train_continual


class TensorFixtureDataset:
    def __init__(self):
        self.targets = [0, 0, 1, 1, 2, 2, 3, 3]
        self._features = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.9, 0.1]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.1, 0.9]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.8, 0.2]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.2, 0.8]),
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self._features[index], self.targets[index]


def _build_stream():
    return SplitCIFAR100TaskStream.from_dataset(
        dataset=TensorFixtureDataset(),
        task_count=2,
        classes_per_task=2,
        split="train",
        sample_id_offset=0,
        class_order=(0, 1, 2, 3),
    )


def test_train_continual_emits_seen_task_accuracy_matrix():
    torch.manual_seed(0)
    stream = _build_stream()
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    result = train_continual(
        model=model,
        train_stream=stream,
        eval_stream=stream,
        optimizer=optimizer,
        config=ContinualTrainerConfig(
            epochs_per_task=8,
            batch_size=4,
            eval_batch_size=4,
            seed=11,
            shuffle_train=True,
        ),
    )

    assert result.task_count == 2
    assert len(result.train_losses) > 0
    assert result.training_time_seconds >= 0.0
    assert result.accuracy_matrix[0][0] is not None
    assert result.accuracy_matrix[0][1] is None
    assert result.accuracy_matrix[1][0] is not None
    assert result.accuracy_matrix[1][1] is not None
    assert result.accuracy_matrix[1][0] >= 0.5
    assert result.accuracy_matrix[1][1] >= 0.5
