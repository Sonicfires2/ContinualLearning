from pathlib import Path

from src.research_protocol import load_research_protocol, summarize_research_protocol


def test_core_protocol_loads_and_matches_locked_scope():
    protocol_path = Path("configs/protocols/core_experiment.yaml")

    protocol = load_research_protocol(protocol_path)

    assert protocol.metadata.protocol_id == "core_split_cifar100_v2"
    assert protocol.scope.required_benchmark == "split_cifar100"
    assert protocol.scope.required_methods == [
        "fine_tuning",
        "random_replay",
        "fixed_periodic_replay",
        "spaced_replay",
    ]
    assert protocol.evaluation.required_metrics == [
        "final_accuracy",
        "average_forgetting",
    ]


def test_protocol_summary_mentions_core_requirements():
    protocol = load_research_protocol("configs/protocols/core_experiment.yaml")

    summary = summarize_research_protocol(protocol)

    assert "split_cifar100" in summary
    assert "fine_tuning, random_replay, fixed_periodic_replay, spaced_replay" in summary
    assert "final_accuracy, average_forgetting" in summary
