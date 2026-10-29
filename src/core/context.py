from __future__ import annotations

import math

from ..models import AgentContext, NodeSnapshot, StarkConfig
from .tree import TreeMemory
from ..utils import changed_anchor_excerpts, last_log_excerpt, normalized_code_hash


def snapshot_node(tree: TreeMemory, node_id: str) -> NodeSnapshot:
    node = tree.get_node(node_id)
    root = tree.get_node(tree.root_id)
    parent = tree.get_node(node.parent_id) if node.parent_id is not None else None
    delta_vs_root = None
    if node.runtime is not None and root.runtime is not None and math.isfinite(node.runtime) and math.isfinite(root.runtime):
        delta_vs_root = node.runtime - root.runtime
    delta_vs_parent = None
    if parent is not None and node.runtime is not None and parent.runtime is not None:
        if math.isfinite(node.runtime) and math.isfinite(parent.runtime):
            delta_vs_parent = node.runtime - parent.runtime
    return NodeSnapshot(
        node_id=node.node_id,
        parent_id=node.parent_id,
        depth=node.depth,
        score=node.score if math.isfinite(node.score) else None,
        status=node.status,
        plan_strategy_name=node.plan_strategy_name,
        failure_type=node.failure_type,
        child_count=len(node.child_ids),
        origin=node.origin,
        selected_count=node.selected_count,
        runtime=node.runtime if node.runtime is not None and math.isfinite(node.runtime) else None,
        latest_failure_stage=node.latest_failure_stage,
        reference_runtime=node.reference_runtime if node.reference_runtime is not None and math.isfinite(node.reference_runtime) else None,
        speedup=node.speedup if node.speedup is not None and math.isfinite(node.speedup) else None,
        delta_vs_root=delta_vs_root,
        delta_vs_parent=delta_vs_parent,
        failure_log_excerpt=last_log_excerpt(node.logs),
        code_hash=normalized_code_hash(node.code),
        plan_summary=node.plan_summary,
        anchor_edits=[
            {"anchor_name": edit.anchor_name, "instruction": edit.instruction, "operation": edit.operation}
            for edit in node.anchor_edits
        ],
        changed_anchor_excerpts=changed_anchor_excerpts(
            root.code,
            node.code,
            [edit.anchor_name for edit in node.anchor_edits],
        ),
        memory_lesson=_memory_lesson(node),
    )


def _memory_lesson(node) -> str | None:
    strategy = node.plan_strategy_name or "this strategy"
    if node.correct:
        if node.speedup is not None and node.speedup >= 1.0:
            return f"REUSE: {strategy} was correct and achieved {node.speedup:.3f}x speedup. Preserve its working edits."
        return f"CORRECT_BUT_SLOW: {strategy} was correct; retain correctness but seek a faster mapping."
    if node.failure_type == "broken_anchor_markers":
        return "AVOID: scaffold text or markers were changed. Edit anchor bodies only and restore the parent scaffold."
    if node.latest_failure_stage == "runtime":
        return f"AVOID_OR_REPAIR: {strategy} failed at runtime. Check int64_t flattened indexing, bounds, shapes, launch sizes, and synchronization."
    if node.latest_failure_stage == "correctness":
        return f"REPAIR: {strategy} compiled but was numerically wrong. Preserve the interface and recheck formulas, masks, reductions, and layout."
    if node.latest_failure_stage == "compile":
        return f"REPAIR: {strategy} did not compile. Recheck C++/CUDA signatures, pybind exports, includes, and types."
    return None


def _dedupe(items: list[NodeSnapshot]) -> list[NodeSnapshot]:
    seen: set[str] = set()
    ordered: list[NodeSnapshot] = []
    for item in items:
        if item.node_id in seen:
            continue
        seen.add(item.node_id)
        ordered.append(item)
    return ordered


def _dedupe_distinct_kernels(items: list[NodeSnapshot]) -> list[NodeSnapshot]:
    seen_hashes: set[str] = set()
    ordered: list[NodeSnapshot] = []
    for item in items:
        key = item.code_hash or item.node_id
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        ordered.append(item)
    return ordered


def _stage_rank(stage: str | None) -> int:
    order = {
        None: 0,
        "compile": 1,
        "runtime": 2,
        "correctness": 3,
    }
    return order.get(stage, 4)


def _sort_snapshots(items: list[NodeSnapshot]) -> list[NodeSnapshot]:
    return sorted(
        items,
        key=lambda item: (
            float("inf") if item.score is None else item.score,
            item.depth,
            _stage_rank(item.latest_failure_stage),
            item.selected_count,
            item.node_id,
        ),
    )


def _limit(items: list[NodeSnapshot], limit: int) -> list[NodeSnapshot]:
    if limit <= 0:
        return []
    return items[:limit]


def build_plan_context(tree: TreeMemory, node_id: str, config: StarkConfig, long_term_memory=None,
                       performance_feedback=None, workflow_guidance=None,
                       operator_memory_evidence=None) -> AgentContext:
    current = snapshot_node(tree, node_id)
    root = snapshot_node(tree, tree.root_id)
    node = tree.get_node(node_id)
    related = [snapshot_node(tree, child_id) for child_id in node.child_ids]
    subtree = tree.subtree_ids(node_id)
    leaders = [
        snapshot_node(tree, leader_id)
        for leader_id in tree.leaderboard
        if leader_id not in subtree
    ]
    return AgentContext(
        role="plan",
        current=current,
        root=root,
        related=_limit(_sort_snapshots(_dedupe(related)), config.context_limit),
        leaders=_limit(_dedupe_distinct_kernels(_sort_snapshots(_dedupe(leaders))), config.context_limit),
        failure=None,
        long_term=list(long_term_memory or []),
        performance_feedback=dict(performance_feedback or {}),
        workflow_guidance=dict(workflow_guidance or {}),
        operator_memory=list(operator_memory_evidence or []),
    )


def build_code_context(tree: TreeMemory, node_id: str, config: StarkConfig, long_term_memory=None) -> AgentContext:
    current = snapshot_node(tree, node_id)
    root = snapshot_node(tree, tree.root_id)
    node = tree.get_node(node_id)
    related_ids = list(node.child_ids) + tree.cousin_child_ids(node_id)
    related = [snapshot_node(tree, related_id) for related_id in related_ids]
    return AgentContext(
        role="code",
        current=current,
        root=root,
        related=_limit(_sort_snapshots(_dedupe(related)), config.context_limit),
        leaders=[],
        failure=None,
        long_term=list(long_term_memory or []),
    )


def build_debug_context(tree: TreeMemory, node_id: str, config: StarkConfig, long_term_memory=None) -> AgentContext:
    current = snapshot_node(tree, node_id)
    root = snapshot_node(tree, tree.root_id)
    siblings = [snapshot_node(tree, sibling_id) for sibling_id in tree.sibling_ids(node_id)]
    return AgentContext(
        role="debug",
        current=current,
        root=root,
        related=_limit(_sort_snapshots(_dedupe(siblings)), config.context_limit),
        leaders=[],
        failure=current if current.latest_failure_stage is not None else None,
        long_term=list(long_term_memory or []),
    )
