"""Evaluate forgetting predictors for saved NLP continual-learning runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.predictors.nlp_forgetting import save_nlp_forgetting_report


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _best_heuristic(report: dict[str, Any]) -> tuple[str | None, float | None]:
    best_name = None
    best_ap = None
    for name, metrics in report["heuristic_report"]["heuristics"].items():
        ap = metrics.get("average_precision")
        if ap is None:
            continue
        if best_ap is None or ap > best_ap:
            best_name = name
            best_ap = float(ap)
    return best_name, best_ap


def _summary_row(run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    best_heuristic_name, best_heuristic_ap = _best_heuristic(report)
    logistic = report["logistic_report"]
    logistic_metrics = logistic.get("metrics") or {}
    return {
        "method": metrics.get("method", run_dir.parent.name),
        "run_dir": str(run_dir),
        "final_accuracy": metrics.get("final_accuracy"),
        "average_forgetting": metrics.get("average_forgetting"),
        "replay_samples": metrics.get("replay_samples"),
        "feature_rows": report["feature_summary"]["row_count"],
        "eligible_rows": report["feature_summary"]["eligible_row_count"],
        "positive_rows": report["feature_summary"]["positive_count"],
        "test_rows": report["heuristic_report"]["test"]["n"],
        "test_positive_rows": report["heuristic_report"]["test"]["positive_count"],
        "test_positive_rate": (
            report["heuristic_report"]["test"]["positive_count"]
            / report["heuristic_report"]["test"]["n"]
            if report["heuristic_report"]["test"]["n"]
            else None
        ),
        "best_heuristic": best_heuristic_name,
        "best_heuristic_ap": best_heuristic_ap,
        "logistic_status": logistic["status"],
        "logistic_reason": logistic.get("reason"),
        "logistic_ap": logistic_metrics.get("average_precision"),
        "logistic_roc_auc": logistic_metrics.get("roc_auc"),
        "logistic_top10_precision": (
            logistic_metrics.get("precision_at_10_percent", {}).get("precision")
            if logistic_metrics
            else None
        ),
        "report_path": str(run_dir / "nlp_forgetting_predictor_report.json"),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Test positive rate | Best heuristic AP | Logistic AP | Logistic ROC-AUC | Top-10% precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['method']} | "
            f"`{row['test_positive_rate']}` | "
            f"`{row['best_heuristic_ap']}` | "
            f"`{row['logistic_ap']}` | "
            f"`{row['logistic_roc_auc']}` | "
            f"`{row['logistic_top10_precision']}` |"
        )
    return "\n".join(lines)


def evaluate_root(root: Path) -> list[dict[str, Any]]:
    signal_paths = sorted(root.glob("*/*/eval_signals.json"))
    if not signal_paths:
        raise FileNotFoundError(f"No eval_signals.json files found under {root}")
    summary_rows: list[dict[str, Any]] = []
    for signal_path in signal_paths:
        run_dir = signal_path.parent
        output_path = run_dir / "nlp_forgetting_predictor_report.json"
        print(f"Building NLP forgetting predictor report for {run_dir}")
        report = save_nlp_forgetting_report(
            signal_path=signal_path,
            output_path=output_path,
        )
        summary_rows.append(_summary_row(run_dir, report))
    _write_json(root / "nlp_forgetting_predictor_summary.json", {"rows": summary_rows})
    (root / "nlp_forgetting_predictor_summary.md").write_text(
        _markdown_table(summary_rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/runs/nlp_continual_pilot"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_root(args.root)
    print("\nSummary:")
    print(_markdown_table(rows))


if __name__ == "__main__":
    main()

