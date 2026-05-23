"""Structured semantic profiles for KernelBench tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OptimizationIntent:
    """A backend-neutral optimization idea attached to a task or anchor."""

    name: str
    summary: str
    target_anchors: list[str] = field(default_factory=list)
    backend_hints: dict[str, list[str]] = field(default_factory=dict)
    risk_notes: list[str] = field(default_factory=list)
    priority: int = 3


@dataclass
class SemanticAnchorProfile:
    """Semantic interpretation for one grounded edit anchor."""

    anchor_name: str
    region_role: str
    semantic_type: str
    source_excerpt: str
    op_names: list[str] = field(default_factory=list)
    optimization_intents: list[str] = field(default_factory=list)
    backend_hints: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    priority: int = 3


@dataclass
class SemanticProfile:
    """Compact task-level semantic summary used by planning prompts."""

    enabled: bool = True
    mode: str = "rule"
    op_type: str = "unknown"
    summary: str = "No semantic pattern was recognized."
    source: str | None = None
    recommended_anchors: list[str] = field(default_factory=list)
    anchors: list[SemanticAnchorProfile] = field(default_factory=list)
    optimization_intents: list[OptimizationIntent] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)


def semantic_profile_to_dict(profile: SemanticProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return asdict(profile)


def semantic_profile_from_dict(payload: dict[str, Any] | None) -> SemanticProfile | None:
    if not payload:
        return None
    anchors = [
        SemanticAnchorProfile(
            anchor_name=str(item.get("anchor_name", "")),
            region_role=str(item.get("region_role", "unknown")),
            semantic_type=str(item.get("semantic_type", "unknown")),
            source_excerpt=str(item.get("source_excerpt", "")),
            op_names=list(item.get("op_names") or []),
            optimization_intents=list(item.get("optimization_intents") or []),
            backend_hints=list(item.get("backend_hints") or []),
            risk_notes=list(item.get("risk_notes") or []),
            priority=int(item.get("priority", 3)),
        )
        for item in payload.get("anchors", [])
        if isinstance(item, dict)
    ]
    intents = [
        OptimizationIntent(
            name=str(item.get("name", "unknown")),
            summary=str(item.get("summary", "")),
            target_anchors=list(item.get("target_anchors") or []),
            backend_hints={str(key): list(value or []) for key, value in dict(item.get("backend_hints") or {}).items()},
            risk_notes=list(item.get("risk_notes") or []),
            priority=int(item.get("priority", 3)),
        )
        for item in payload.get("optimization_intents", [])
        if isinstance(item, dict)
    ]
    return SemanticProfile(
        enabled=bool(payload.get("enabled", True)),
        mode=str(payload.get("mode", "rule")),
        op_type=str(payload.get("op_type", "unknown")),
        summary=str(payload.get("summary", "No semantic pattern was recognized.")),
        source=payload.get("source"),
        recommended_anchors=list(payload.get("recommended_anchors") or []),
        anchors=anchors,
        optimization_intents=intents,
        risk_notes=list(payload.get("risk_notes") or []),
    )
