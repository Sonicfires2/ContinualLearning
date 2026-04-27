from src.data.split_cifar100 import (
    SplitCIFAR100TaskStream,
    build_task_specs,
)


class FixtureClassDataset:
    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return f"sample-{index}", self.targets[index]


def test_task_specs_are_deterministic_for_fixed_seed():
    targets = [class_id for class_id in range(6) for _ in range(2)]

    first = build_task_specs(
        task_count=3,
        classes_per_task=2,
        split_seed=17,
        targets=targets,
    )
    second = build_task_specs(
        task_count=3,
        classes_per_task=2,
        split_seed=17,
        targets=targets,
    )

    assert first == second


def test_task_specs_cover_all_classes_without_overlap():
    specs = build_task_specs(
        task_count=3,
        classes_per_task=2,
        class_order=(0, 1, 2, 3, 4, 5),
    )

    task_classes = [set(spec.class_ids) for spec in specs]
    all_classes = [class_id for spec in specs for class_id in spec.class_ids]

    assert sorted(all_classes) == [0, 1, 2, 3, 4, 5]
    assert len(all_classes) == len(set(all_classes))
    assert task_classes[0].isdisjoint(task_classes[1])
    assert task_classes[0].isdisjoint(task_classes[2])
    assert task_classes[1].isdisjoint(task_classes[2])


def test_task_stream_exposes_expected_counts_and_stable_sample_metadata():
    dataset = FixtureClassDataset(targets=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    stream = SplitCIFAR100TaskStream.from_dataset(
        dataset=dataset,
        task_count=3,
        classes_per_task=2,
        split="train",
        sample_id_offset=0,
        class_order=(0, 1, 2, 3, 4, 5),
    )

    task_zero = stream.task_dataset(0)

    assert len(stream) == 3
    assert [len(stream.task_dataset(task_id)) for task_id in range(3)] == [4, 4, 4]
    assert task_zero[0] == {
        "x": "sample-0",
        "y": 0,
        "sample_id": 0,
        "task_id": 0,
        "original_class_id": 0,
        "within_task_label": 0,
        "original_index": 0,
        "split": "train",
    }
    assert task_zero[2]["original_class_id"] == 1
    assert task_zero[2]["within_task_label"] == 1
    assert task_zero[2]["sample_id"] == 2


def test_split_offsets_keep_train_and_test_sample_ids_distinct():
    dataset = FixtureClassDataset(targets=[0, 0, 1, 1])

    train_stream = SplitCIFAR100TaskStream.from_dataset(
        dataset=dataset,
        task_count=1,
        classes_per_task=2,
        split="train",
        sample_id_offset=0,
        class_order=(0, 1),
    )
    test_stream = SplitCIFAR100TaskStream.from_dataset(
        dataset=dataset,
        task_count=1,
        classes_per_task=2,
        split="test",
        sample_id_offset=1_000_000,
        class_order=(0, 1),
    )

    assert train_stream.task_dataset(0)[0]["sample_id"] == 0
    assert test_stream.task_dataset(0)[0]["sample_id"] == 1_000_000


def test_invalid_task_shapes_and_class_orders_are_rejected():
    try:
        build_task_specs(task_count=0, classes_per_task=2)
    except ValueError as exc:
        assert "task_count" in str(exc)
    else:
        raise AssertionError("expected invalid task_count to be rejected")

    try:
        build_task_specs(
            task_count=2,
            classes_per_task=2,
            class_order=(0, 1, 1, 2),
        )
    except ValueError as exc:
        assert "duplicate class IDs" in str(exc)
    else:
        raise AssertionError("expected duplicate class order to be rejected")
