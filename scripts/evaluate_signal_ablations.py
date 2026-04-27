"""Run feature-group ablations for forgetting-risk predictors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors import save_signal_ablation_report


def _resolve_signal_path(args: argparse.Namespace) -> Path:
    if args.signals is not None:
        return Path(args.signals)
    if args.run_dir is None:
        raise ValueError("provide either --signals or --run-dir")
    return Path(args.run_dir) / "sample_signals.json"


def _resolve_label_path(args: argparse.Namespace) -> Path:
    if args.labels is not None:
        return Path(args.labels)
    if args.run_dir is None:
        raise ValueError("provide either --labels or --run-dir")
    return Path(args.run_dir) / "forgetting_labels.json"


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.run_dir is not None:
        return Path(args.run_dir) / "signal_ablation_report.json"
    return Path(args.labels).with_name("signal_ablation_report.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate signal-group ablations for learned forgetting predictors"
    )
    parser.add_argument("--run-dir", type=str, help="Run artifact directory")
    parser.add_argument("--signals", type=str, help="Path to sample_signals.json")
    parser.add_argument("--labels", type=str, help="Path to forgetting_labels.json")
    parser.add_argument("--output", type=str, help="Output report path")
    args = parser.parse_args(argv)

    try:
        output_path = _resolve_output_path(args)
        report = save_signal_ablation_report(
            signal_path=_resolve_signal_path(args),
            label_path=_resolve_label_path(args),
            output_path=output_path,
        )
    except Exception as exc:
        print(f"Failed to evaluate signal ablations: {exc}", file=sys.stderr)
        return 2

    ablation = report["ablation_report"]
    best_heuristic = ablation["best_heuristic"]
    best_group = (
        ablation["ranked_feature_groups"][0]
        if ablation["ranked_feature_groups"]
        else {"feature_group": None, "best_model": None, "average_precision": None}
    )
    threshold = ablation["all_features_logistic_threshold_recommendation"]
    print(f"Saved signal-ablation report to: {output_path}")
    print(
        "Temporal split: "
        f"train<=task {ablation['temporal_split']['train_anchor_task_max']}, "
        f"test>=task {ablation['temporal_split']['test_anchor_task_min']}"
    )
    print(
        "Best heuristic AP: "
        f"{best_heuristic['name']}={best_heuristic['average_precision']}"
    )
    print(
        "Best feature group AP: "
        f"{best_group['feature_group']}:{best_group['best_model']}="
        f"{best_group['average_precision']}"
    )
    print(
        "Recommended logistic threshold: "
        f"{threshold.get('threshold')} "
        f"(precision={threshold.get('precision')}, recall={threshold.get('recall')})"
    )
    print(
        "Use learned online gate next: "
        f"{ablation['recommendation']['use_learned_online_gate_next']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
