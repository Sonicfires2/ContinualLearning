"""Evaluate whether expensive gradient signals improve forgetting prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors import save_expensive_signal_diagnostic_report


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


def _resolve_gradient_path(args: argparse.Namespace) -> Path:
    if args.gradients is not None:
        return Path(args.gradients)
    if args.run_dir is None:
        raise ValueError("provide either --gradients or --run-dir")
    return Path(args.run_dir) / "gradient_signals.json"


def _resolve_diagnostic_metrics_path(args: argparse.Namespace) -> Path | None:
    if args.diagnostic_metrics is not None:
        return Path(args.diagnostic_metrics)
    if args.run_dir is not None:
        return Path(args.run_dir) / "metrics.json"
    return None


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.run_dir is not None:
        return Path(args.run_dir) / "expensive_signal_diagnostic_report.json"
    return Path(args.gradients).with_name("expensive_signal_diagnostic_report.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate gradient-signal diagnostic value for forgetting prediction"
    )
    parser.add_argument("--run-dir", type=str, help="Run artifact directory")
    parser.add_argument("--signals", type=str, help="Path to sample_signals.json")
    parser.add_argument("--labels", type=str, help="Path to forgetting_labels.json")
    parser.add_argument("--gradients", type=str, help="Path to gradient_signals.json")
    parser.add_argument("--reference-metrics", type=str, help="Reference metrics.json path")
    parser.add_argument("--diagnostic-metrics", type=str, help="Diagnostic metrics.json path")
    parser.add_argument("--output", type=str, help="Output report path")
    args = parser.parse_args(argv)

    try:
        output_path = _resolve_output_path(args)
        report = save_expensive_signal_diagnostic_report(
            signal_path=_resolve_signal_path(args),
            label_path=_resolve_label_path(args),
            gradient_path=_resolve_gradient_path(args),
            output_path=output_path,
            reference_metrics_path=(
                Path(args.reference_metrics) if args.reference_metrics else None
            ),
            diagnostic_metrics_path=_resolve_diagnostic_metrics_path(args),
        )
    except Exception as exc:
        print(f"Failed to evaluate expensive signal diagnostics: {exc}", file=sys.stderr)
        return 2

    recommendation = report["recommendation"]
    ablation = report["ablation_report"]
    ranked = ablation["ranked_feature_groups"]
    best = ranked[0] if ranked else {
        "feature_group": None,
        "best_model": None,
        "average_precision": None,
    }
    print(f"Saved expensive-signal diagnostic report to: {output_path}")
    print(
        "Best feature group AP: "
        f"{best['feature_group']}:{best['best_model']}={best['average_precision']}"
    )
    print(
        "Build replay intervention next: "
        f"{recommendation['build_replay_intervention_next']}"
    )
    print(f"Reason: {recommendation['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
