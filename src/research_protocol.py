"""Utilities for loading and validating locked research protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class ProtocolMetadata:
    protocol_id: str = ""
    version: int = 1
    status: str = "draft"
    locked_on: str = ""
    owner: str = ""
    rationale: str = ""


@dataclass
class ResearchSection:
    goal: str = ""
    question: str = ""


@dataclass
class ScopeSection:
    required_benchmark: str = ""
    task_count: int = 0
    classes_per_task: int = 0
    required_methods: list[str] = field(default_factory=list)
    recommended_methods: list[str] = field(default_factory=list)
    stretch_methods: list[str] = field(default_factory=list)
    required_signals: list[str] = field(default_factory=list)
    optional_signals: list[str] = field(default_factory=list)
    required_telemetry: list[str] = field(default_factory=list)
    optional_benchmarks: list[str] = field(default_factory=list)


@dataclass
class EvaluationSection:
    fixed_controls: list[str] = field(default_factory=list)
    required_metrics: list[str] = field(default_factory=list)
    recommended_metrics: list[str] = field(default_factory=list)
    required_artifacts: list[str] = field(default_factory=list)
    evaluation_schedule: str = ""


@dataclass
class ScopeGuardsSection:
    out_of_core_scope: list[str] = field(default_factory=list)
    promotion_rule: str = ""


@dataclass
class ResearchProtocol:
    metadata: ProtocolMetadata = field(default_factory=ProtocolMetadata)
    research: ResearchSection = field(default_factory=ResearchSection)
    scope: ScopeSection = field(default_factory=ScopeSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    scope_guards: ScopeGuardsSection = field(default_factory=ScopeGuardsSection)


def _find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _overlap(left: list[str], right: list[str]) -> list[str]:
    return sorted(set(left).intersection(right))


def validate_research_protocol(
    protocol: ResearchProtocol,
    *,
    source: Path | None = None,
) -> None:
    """Raise a ValueError if the protocol is incomplete or internally inconsistent."""

    errors: list[str] = []

    if not protocol.metadata.protocol_id:
        errors.append("metadata.protocol_id must be set")
    if protocol.metadata.version < 1:
        errors.append("metadata.version must be at least 1")
    if not protocol.research.goal.strip():
        errors.append("research.goal must be set")
    if not protocol.research.question.strip():
        errors.append("research.question must be set")
    if not protocol.scope.required_benchmark:
        errors.append("scope.required_benchmark must be set")
    if protocol.scope.task_count < 1:
        errors.append("scope.task_count must be greater than 0")
    if protocol.scope.classes_per_task < 1:
        errors.append("scope.classes_per_task must be greater than 0")
    if not protocol.scope.required_methods:
        errors.append("scope.required_methods must contain at least one method")
    if not protocol.scope.required_signals:
        errors.append("scope.required_signals must contain at least one signal")
    if not protocol.evaluation.fixed_controls:
        errors.append("evaluation.fixed_controls must contain at least one control")
    if not protocol.evaluation.required_metrics:
        errors.append("evaluation.required_metrics must contain at least one metric")
    if not protocol.evaluation.required_artifacts:
        errors.append("evaluation.required_artifacts must contain at least one artifact")
    if not protocol.evaluation.evaluation_schedule:
        errors.append("evaluation.evaluation_schedule must be set")
    if not protocol.scope_guards.promotion_rule.strip():
        errors.append("scope_guards.promotion_rule must be set")

    duplicate_groups = {
        "scope.required_methods": _find_duplicates(protocol.scope.required_methods),
        "scope.recommended_methods": _find_duplicates(protocol.scope.recommended_methods),
        "scope.stretch_methods": _find_duplicates(protocol.scope.stretch_methods),
        "scope.required_signals": _find_duplicates(protocol.scope.required_signals),
        "scope.optional_signals": _find_duplicates(protocol.scope.optional_signals),
        "scope.required_telemetry": _find_duplicates(protocol.scope.required_telemetry),
        "scope.optional_benchmarks": _find_duplicates(protocol.scope.optional_benchmarks),
        "evaluation.fixed_controls": _find_duplicates(protocol.evaluation.fixed_controls),
        "evaluation.required_metrics": _find_duplicates(protocol.evaluation.required_metrics),
        "evaluation.recommended_metrics": _find_duplicates(protocol.evaluation.recommended_metrics),
        "evaluation.required_artifacts": _find_duplicates(protocol.evaluation.required_artifacts),
    }
    for group_name, duplicates in duplicate_groups.items():
        if duplicates:
            errors.append(f"{group_name} contains duplicates: {', '.join(duplicates)}")

    method_overlap = _overlap(
        protocol.scope.required_methods,
        protocol.scope.recommended_methods + protocol.scope.stretch_methods,
    )
    if method_overlap:
        errors.append(
            "method tiers overlap across required/recommended/stretch: "
            + ", ".join(method_overlap)
        )

    signal_overlap = _overlap(protocol.scope.required_signals, protocol.scope.optional_signals)
    if signal_overlap:
        errors.append(
            "signal tiers overlap across required/optional: " + ", ".join(signal_overlap)
        )

    if errors:
        location = f" in {source}" if source is not None else ""
        raise ValueError("Invalid research protocol" + location + ": " + "; ".join(errors))


def load_research_protocol(path: str | Path) -> ResearchProtocol:
    """Load a versioned research protocol from YAML and validate it."""

    resolved_path = Path(path)
    raw_config = OmegaConf.load(resolved_path)
    merged_config = OmegaConf.merge(OmegaConf.structured(ResearchProtocol), raw_config)
    protocol = OmegaConf.to_object(merged_config)
    validate_research_protocol(protocol, source=resolved_path)
    return protocol


def summarize_research_protocol(protocol: ResearchProtocol) -> str:
    """Return a compact human-readable summary of a loaded protocol."""

    return "\n".join(
        [
            f"Protocol: {protocol.metadata.protocol_id} (v{protocol.metadata.version}, {protocol.metadata.status})",
            f"Primary benchmark: {protocol.scope.required_benchmark}",
            (
                "Required methods: "
                + ", ".join(protocol.scope.required_methods)
            ),
            (
                "Required signals: "
                + ", ".join(protocol.scope.required_signals)
            ),
            (
                "Required metrics: "
                + ", ".join(protocol.evaluation.required_metrics)
            ),
            f"Evaluation schedule: {protocol.evaluation.evaluation_schedule}",
        ]
    )
