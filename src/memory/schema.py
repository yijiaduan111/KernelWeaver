from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryMethodCard:
    method_id: str
    title: str
    summary: str
    source: str = "km_bottleneck"
    target_anchors: list[str] = field(default_factory=list)
    why_now: list[str] = field(default_factory=list)
    implementation_hints: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    expected_metric_change: list[str] = field(default_factory=list)
    priority: int = 3


@dataclass
class MemoryProfile:
    enabled: bool = True
    mode: str = "expert_memory_v0"
    backend: str = "cuda"
    op_type: str = "unknown"
    source: str = "machine_check"
    bottleneck_id: str | None = None
    case_id: str | None = None
    allowed_methods: list[str] = field(default_factory=list)
    forbidden_methods: list[str] = field(default_factory=list)
    bootstrap_cards: list[MemoryMethodCard] = field(default_factory=list)
    challenger_cards: list[MemoryMethodCard] = field(default_factory=list)
    preferred_methods: list[str] = field(default_factory=list)
    blocked_methods: list[str] = field(default_factory=list)
    feedback_digest: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def memory_profile_to_dict(profile: MemoryProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return asdict(profile)


def memory_profile_from_dict(payload: dict[str, Any] | None) -> MemoryProfile | None:
    if not payload:
        return None
    return MemoryProfile(
        enabled=bool(payload.get("enabled", True)),
        mode=str(payload.get("mode", "expert_memory_v0")),
        backend=str(payload.get("backend", "cuda")),
        op_type=str(payload.get("op_type", "unknown")),
        source=str(payload.get("source", "machine_check")),
        bottleneck_id=str(payload.get("bottleneck_id", "")) or None,
        case_id=str(payload.get("case_id", "")) or None,
        allowed_methods=[str(item) for item in payload.get("allowed_methods", []) if str(item).strip()],
        forbidden_methods=[str(item) for item in payload.get("forbidden_methods", []) if str(item).strip()],
        bootstrap_cards=[_card_from_dict(item) for item in payload.get("bootstrap_cards", []) if isinstance(item, dict)],
        challenger_cards=[_card_from_dict(item) for item in payload.get("challenger_cards", []) if isinstance(item, dict)],
        preferred_methods=[str(item) for item in payload.get("preferred_methods", []) if str(item).strip()],
        blocked_methods=[str(item) for item in payload.get("blocked_methods", []) if str(item).strip()],
        feedback_digest=dict(payload.get("feedback_digest") or {}),
        notes=[str(item) for item in payload.get("notes", []) if str(item).strip()],
    )


def _card_from_dict(payload: dict[str, Any]) -> MemoryMethodCard:
    return MemoryMethodCard(
        method_id=str(payload.get("method_id", "")),
        title=str(payload.get("title", "")),
        summary=str(payload.get("summary", "")),
        source=str(payload.get("source", "km_bottleneck")),
        target_anchors=[str(item) for item in payload.get("target_anchors", []) if str(item).strip()],
        why_now=[str(item) for item in payload.get("why_now", []) if str(item).strip()],
        implementation_hints=[str(item) for item in payload.get("implementation_hints", []) if str(item).strip()],
        forbidden_patterns=[str(item) for item in payload.get("forbidden_patterns", []) if str(item).strip()],
        expected_metric_change=[str(item) for item in payload.get("expected_metric_change", []) if str(item).strip()],
        priority=int(payload.get("priority", 3)),
    )
