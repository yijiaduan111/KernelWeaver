from __future__ import annotations

import math
import re

from ..feedback.schema import FeedbackState
from ..models import AgentContext, NodeSnapshot, StarkConfig
from .anchor_policy import compute_anchor_policy
from .regions import extract_region_names
from .tree import TreeMemory
from ..utils import last_log_excerpt, normalized_code_hash


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
    )


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


def _ancestors(tree: TreeMemory, node_id: str) -> set[str]:
    ancestors: set[str] = set()
    current = tree.get_node(node_id)
    while current.parent_id is not None:
        ancestors.add(current.parent_id)
        current = tree.get_node(current.parent_id)
    return ancestors


def _best_node_id(tree: TreeMemory, node_id: str) -> str | None:
    candidates = [leader_id for leader_id in tree.leaderboard if leader_id != node_id]
    if candidates:
        return candidates[0]
    return tree.root_id if tree.root_id != node_id else None


def _extract_region_excerpt(source_code: str, region: str, max_chars: int = 1200) -> str | None:
    pattern = re.compile(
        r"(?ms)^[ \t]*#\s*<<<IMPROVE:" + re.escape(region) + r">>>(?:\r?\n)(?P<body>.*?)(^[ \t]*#\s*<<<END_IMPROVE>>>)"
    )
    match = pattern.search(source_code)
    if not match:
        return None
    body = match.group("body").strip("\n")
    return body[:max_chars]


def _build_best_kernel_summary(snapshot: NodeSnapshot | None, code: str | None) -> dict | None:
    if snapshot is None or code is None:
        return None
    return {
        "node_id": snapshot.node_id,
        "speedup": snapshot.speedup,
        "plan_strategy_name": snapshot.plan_strategy_name,
        "status": snapshot.status,
        "edited_anchors": extract_region_names(code),
    }


def _context_best_kernel(
    tree: TreeMemory,
    task,
    node_id: str,
    feedback_state: FeedbackState | None,
) -> tuple[NodeSnapshot | None, str | None, dict[str, str], dict | None, list[str], list[str]]:
    best_id = _best_node_id(tree, node_id)
    if best_id is None:
        return None, None, {}, None, [], []
    best_node = tree.get_node(best_id)
    best_snapshot = snapshot_node(tree, best_id)
    best_code = best_node.code
    policy = compute_anchor_policy(task, tree.get_node(node_id), feedback_state)
    excerpts = {
        anchor: excerpt
        for anchor in policy.active
        if (excerpt := _extract_region_excerpt(best_code, anchor))
    }
    summary = _build_best_kernel_summary(best_snapshot, best_code)
    return best_snapshot, best_code, excerpts, summary, list(policy.active), list(policy.frozen)


def build_plan_context(tree: TreeMemory, task, node_id: str, config: StarkConfig, feedback_state: FeedbackState | None = None) -> AgentContext:
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
    best_node, best_code, best_excerpt, best_summary, active_anchors, frozen_anchors = _context_best_kernel(
        tree,
        task,
        node_id,
        feedback_state,
    )
    return AgentContext(
        role="plan",
        current=current,
        root=root,
        related=_limit(_sort_snapshots(_dedupe(related)), config.context_limit),
        leaders=_limit(_dedupe_distinct_kernels(_sort_snapshots(_dedupe(leaders))), config.context_limit),
        failure=None,
        feedback_state=feedback_state,
        best_node=best_node,
        best_kernel_code=best_code,
        best_kernel_excerpt=best_excerpt,
        best_kernel_summary=best_summary,
        active_anchors=active_anchors,
        frozen_anchors=frozen_anchors,
    )


def build_code_context(tree: TreeMemory, task, node_id: str, config: StarkConfig, feedback_state: FeedbackState | None = None) -> AgentContext:
    current = snapshot_node(tree, node_id)
    root = snapshot_node(tree, tree.root_id)
    node = tree.get_node(node_id)
    related_ids = list(node.child_ids) + tree.cousin_child_ids(node_id)
    related = [snapshot_node(tree, related_id) for related_id in related_ids]
    current_ancestors = _ancestors(tree, node_id)
    code_leaders = [
        snapshot_node(tree, leader_id)
        for leader_id in tree.leaderboard
        if leader_id != node_id and leader_id not in current_ancestors
    ]
    best_node, best_code, best_excerpt, best_summary, active_anchors, frozen_anchors = _context_best_kernel(
        tree,
        task,
        node_id,
        feedback_state,
    )
    return AgentContext(
        role="code",
        current=current,
        root=root,
        related=_limit(_sort_snapshots(_dedupe(related)), config.context_limit),
        leaders=_limit(_dedupe_distinct_kernels(_sort_snapshots(_dedupe(code_leaders))), 2),
        failure=None,
        feedback_state=feedback_state,
        best_node=best_node,
        best_kernel_code=best_code,
        best_kernel_excerpt=best_excerpt,
        best_kernel_summary=best_summary,
        active_anchors=active_anchors,
        frozen_anchors=frozen_anchors,
    )


def build_debug_context(tree: TreeMemory, task, node_id: str, config: StarkConfig, feedback_state: FeedbackState | None = None) -> AgentContext:
    current = snapshot_node(tree, node_id)
    root = snapshot_node(tree, tree.root_id)
    siblings = [snapshot_node(tree, sibling_id) for sibling_id in tree.sibling_ids(node_id)]
    best_node, best_code, best_excerpt, best_summary, active_anchors, frozen_anchors = _context_best_kernel(
        tree,
        task,
        node_id,
        feedback_state,
    )
    return AgentContext(
        role="debug",
        current=current,
        root=root,
        related=_limit(_sort_snapshots(_dedupe(siblings)), config.context_limit),
        leaders=[],
        failure=current if current.latest_failure_stage is not None else None,
        feedback_state=feedback_state,
        best_node=best_node,
        best_kernel_code=best_code,
        best_kernel_excerpt=best_excerpt,
        best_kernel_summary=best_summary,
        active_anchors=active_anchors,
        frozen_anchors=frozen_anchors,
    )
