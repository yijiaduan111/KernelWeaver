"""Prompt rendering helpers for semantic profiles."""

from __future__ import annotations

from typing import Any

from .schema import SemanticProfile


def semantic_profile_to_prompt_dict(profile: SemanticProfile | None, max_anchor_hints: int | None = None) -> dict[str, Any] | None:
    if profile is None or not profile.enabled:
        return None
    limit = max_anchor_hints or len(profile.anchors)
    return {
        "mode": profile.mode,
        "op_type": profile.op_type,
        "summary": profile.summary,
        "workload_tag": profile.workload_tag,
        "bottleneck_hint": profile.bottleneck_hint,
        "recommended_anchors": list(profile.recommended_anchors[:limit]),
        "optimization_intents": [
            {
                "name": intent.name,
                "summary": intent.summary,
                "target_anchors": list(intent.target_anchors[:limit]),
                "backend_hints": intent.backend_hints,
                "risk_notes": intent.risk_notes,
            }
            for intent in profile.optimization_intents[:3]
        ],
        "anchor_hints": [
            {
                "anchor_name": anchor.anchor_name,
                "region_role": anchor.region_role,
                "semantic_type": anchor.semantic_type,
                "op_names": anchor.op_names[:8],
                "optimization_intents": anchor.optimization_intents[:4],
                "backend_hints": anchor.backend_hints[:4],
                "risk_notes": anchor.risk_notes[:3],
            }
            for anchor in profile.anchors[:limit]
        ],
        "risk_notes": profile.risk_notes[:5],
    }
