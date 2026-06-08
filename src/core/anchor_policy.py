from __future__ import annotations

from dataclasses import dataclass, field

from ..feedback.schema import FeedbackState
from ..models import SearchNode, TaskSpec


@dataclass(frozen=True)
class AnchorPolicyDecision:
    active: list[str] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)


_PREFERRED_REGION_NAMES = (
    "cuda_cu",
    "tilelang_kernel",
    "cute_kernel",
    "triton_kernel",
    "forward_body",
)

_LOW_RISK_MUTABLE_REGIONS = {
    "cuda_cpp",
    "init_body",
}

_ALWAYS_FROZEN_REGIONS = {
    "user_helpers",
    "helpers",
}


def compute_anchor_policy(
    task: TaskSpec,
    parent_node: SearchNode,
    feedback_state: FeedbackState | None,
) -> AnchorPolicyDecision:
    anchors = [region.anchor_name for region in task.grounded_regions]
    if not anchors:
        return AnchorPolicyDecision()

    if not _should_refine(parent_node, feedback_state):
        return AnchorPolicyDecision(active=list(anchors), frozen=[])

    active = [name for name in anchors if _is_preferred_active(name)]
    if not active:
        active = [name for name in anchors if name in _LOW_RISK_MUTABLE_REGIONS]
    if not active:
        active = list(anchors)
    frozen = [name for name in anchors if name not in active or name in _ALWAYS_FROZEN_REGIONS]
    return AnchorPolicyDecision(active=active, frozen=frozen)


def _should_refine(parent_node: SearchNode, feedback_state: FeedbackState | None) -> bool:
    if parent_node.speedup is not None and parent_node.speedup > 1.5:
        return True
    if feedback_state is None:
        return False
    return feedback_state.phase == "refinement" and (feedback_state.best_speedup or 0.0) > 1.5


def _is_preferred_active(name: str) -> bool:
    if name in _ALWAYS_FROZEN_REGIONS:
        return False
    if name in _LOW_RISK_MUTABLE_REGIONS:
        return True
    if name.startswith("forward_stmt_"):
        return False
    return name in _PREFERRED_REGION_NAMES or name.endswith("_kernel")
