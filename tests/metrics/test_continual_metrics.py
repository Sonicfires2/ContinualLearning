from src.metrics.continual import (
    average_accuracy_after_task,
    average_forgetting,
    final_accuracy,
    forgetting_by_task,
)


def test_continual_metrics_match_locked_definitions():
    accuracy_matrix = [
        [0.90, None, None],
        [0.70, 0.80, None],
        [0.60, 0.75, 0.85],
    ]

    assert average_accuracy_after_task(accuracy_matrix, 0) == 0.90
    assert final_accuracy(accuracy_matrix) == (0.60 + 0.75 + 0.85) / 3
    assert forgetting_by_task(accuracy_matrix) == {
        0: 0.90 - 0.60,
        1: 0.80 - 0.75,
    }
    assert average_forgetting(accuracy_matrix) == ((0.90 - 0.60) + (0.80 - 0.75)) / 2
