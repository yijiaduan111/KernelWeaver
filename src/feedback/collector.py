from __future__ import annotations

from collections import Counter, defaultdict

from ..core.tree import TreeMemory
from ..models import SearchNode
from .schema import AttemptRecord, FeedbackState, StrategyAttemptSummary


def _node_order_key(node: SearchNode) -> tuple[int, str]:
    if node.node_id == "root":
        return (0, "root")
    if node.node_id.startswith("n") and node.node_id[1:].isdigit():
        return (1, f"{int(node.node_id[1:]):08d}")
    return (2, node.node_id)


def _attempt_from_node(tree: TreeMemory, node: SearchNode) -> AttemptRecord | None:
    if node.node_id == tree.root_id:
        return None
    parent = tree.get_node(node.parent_id) if node.parent_id is not None else None
    return AttemptRecord(
        attempt_id=node.node_id,
        parent_id=node.parent_id,
        strategy_name=node.plan_strategy_name,
        anchor_names=[edit.anchor_name for edit in node.anchor_edits],
        origin=node.origin,
        compile_ok=node.compile_ok,
        correct=node.correct,
        speedup=node.speedup,
        failure_stage=node.latest_failure_stage,
        failure_type=node.failure_type,
        parent_speedup=parent.speedup if parent is not None else None,
        node_status=node.node_status,
    )


def _build_strategy_summaries(attempts: list[AttemptRecord]) -> list[StrategyAttemptSummary]:
    grouped: dict[str, list[AttemptRecord]] = defaultdict(list)
    for attempt in attempts:
        if attempt.strategy_name:
            grouped[attempt.strategy_name].append(attempt)

    summaries: list[StrategyAttemptSummary] = []
    for strategy_name, items in grouped.items():
        deltas = [
            item.speedup - item.parent_speedup
            for item in items
            if item.speedup is not None and item.parent_speedup is not None
        ]
        failures = [item.failure_type for item in items if item.failure_type]
        summaries.append(
            StrategyAttemptSummary(
                strategy_name=strategy_name,
                total_attempts=len(items),
                compile_success=sum(1 for item in items if item.compile_ok),
                correct_success=sum(1 for item in items if item.correct),
                best_speedup=max((item.speedup for item in items if item.speedup is not None), default=None),
                avg_delta_vs_parent=(sum(deltas) / len(deltas)) if deltas else None,
                dominant_failure_type=Counter(failures).most_common(1)[0][0] if failures else None,
            )
        )
    return sorted(
        summaries,
        key=lambda item: (
            1 if item.best_speedup is None else 0,
            -(item.best_speedup or 0.0),
            item.strategy_name,
        ),
    )


def _infer_phase(attempts: list[AttemptRecord]) -> str:
    if not attempts:
        return "exploration"
    correct_count = sum(1 for attempt in attempts if attempt.correct)
    if correct_count == 0:
        return "exploration"
    best = max((attempt.speedup for attempt in attempts if attempt.speedup is not None), default=None)
    if best is None or best < 1.05:
        return "exploitation"
    recent_correct = [attempt for attempt in attempts[-5:] if attempt.correct and attempt.speedup is not None]
    if len(recent_correct) >= 2:
        deltas = [attempt.speedup - (attempt.parent_speedup or 1.0) for attempt in recent_correct]
        if max(deltas) < 0.1:
            return "refinement"
    return "refinement" if best > 1.5 else "exploitation"


def collect_feedback_state(tree: TreeMemory, recent_window: int = 8, failure_window: int = 5) -> FeedbackState:
    ordered_nodes = sorted(tree.nodes.values(), key=_node_order_key)
    attempts = [attempt for node in ordered_nodes if (attempt := _attempt_from_node(tree, node)) is not None]
    total_attempts = len(attempts)
    failed_attempts = [attempt for attempt in attempts if attempt.failure_type]
    best_attempt = max(
        (attempt for attempt in attempts if attempt.speedup is not None),
        key=lambda attempt: attempt.speedup or float("-inf"),
        default=None,
    )
    return FeedbackState(
        strategy_summaries=_build_strategy_summaries(attempts),
        total_attempts=total_attempts,
        compile_rate=(sum(1 for attempt in attempts if attempt.compile_ok) / total_attempts) if total_attempts else 0.0,
        correct_rate=(sum(1 for attempt in attempts if attempt.correct) / total_attempts) if total_attempts else 0.0,
        best_speedup=best_attempt.speedup if best_attempt else None,
        best_strategy_name=best_attempt.strategy_name if best_attempt else None,
        recent_failure_types=[attempt.failure_type for attempt in failed_attempts[-failure_window:]],
        recent_attempts=attempts[-recent_window:],
        phase=_infer_phase(attempts),
    )
