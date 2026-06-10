from __future__ import annotations

from typing import Any

from .schema import MemoryMethodCard, MemoryProfile


def memory_profile_to_prompt_dict(profile: MemoryProfile | None, max_cards: int | None = None) -> dict[str, Any] | None:
    if profile is None or not profile.enabled:
        return None
    limit = max_cards or max(len(profile.bootstrap_cards), 1)
    return {
        "mode": profile.mode,
        "backend": profile.backend,
        "op_type": profile.op_type,
        "source": profile.source,
        "bottleneck_id": profile.bottleneck_id,
        "case_id": profile.case_id,
        "allowed_methods": list(profile.allowed_methods[: max(limit, 4)]),
        "forbidden_methods": list(profile.forbidden_methods[: max(limit, 4)]),
        "primary_methods": [_card_to_prompt_dict(card) for card in profile.bootstrap_cards[:limit]],
        "challenger_methods": [_card_to_prompt_dict(card) for card in profile.challenger_cards[:limit]],
        "preferred_methods": list(profile.preferred_methods[:limit]),
        "blocked_methods": list(profile.blocked_methods[:limit]),
        "feedback_digest": dict(profile.feedback_digest),
        "notes": list(profile.notes[:6]),
    }


def _card_to_prompt_dict(card: MemoryMethodCard) -> dict[str, Any]:
    return {
        "method_id": card.method_id,
        "title": card.title,
        "summary": card.summary,
        "target_anchors": list(card.target_anchors[:4]),
        "why_now": list(card.why_now[:4]),
        "implementation_hints": list(card.implementation_hints[:6]),
        "forbidden_patterns": list(card.forbidden_patterns[:4]),
        "expected_metric_change": list(card.expected_metric_change[:4]),
        "priority": card.priority,
    }
