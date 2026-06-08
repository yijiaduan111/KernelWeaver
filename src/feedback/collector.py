from __future__ import annotations

from ..core.tree import TreeMemory
from ..models import SearchNode
from .schema import AttemptRecord, ChampionState, FeedbackState


def _node_order_key(node: SearchNode) -> tuple[int, str]:
    if node.node_id == "root":
        return (0, "root")
    if node.node_id.startswith("n") and node.node_id[1:].isdigit():
        return (1, f"{int(node.node_id[1:]):08d}")
    return (2, node.node_id)


def _lineage(tree: TreeMemory, node_id: str | None) -> list[str]:
    if node_id is None or node_id not in tree.nodes:
        return []
    chain: list[str] = []
    current = tree.get_node(node_id)
    while True:
        chain.append(current.node_id)
        if current.parent_id is None:
            break
        current = tree.get_node(current.parent_id)
    chain.reverse()
    return chain


def _attempt_from_node(tree: TreeMemory, node: SearchNode) -> AttemptRecord | None:
    if node.node_id == tree.root_id:
        return None
    parent = tree.get_node(node.parent_id) if node.parent_id is not None else None
    return AttemptRecord(
        attempt_id=node.node_id,
        parent_id=node.parent_id,
        strategy_name=node.plan_strategy_name,
        mode=node.plan_mode or "explore",
        mutation_family=node.mutation_family,
        speedup=node.speedup,
        parent_speedup=parent.speedup if parent is not None else None,
        compile_ok=node.compile_ok,
        correct=node.correct,
        failure_stage=node.latest_failure_stage,
        failure_type=node.failure_type,
        single_change_focus=node.single_change_focus,
    )


def collect_feedback_state(tree: TreeMemory, plateau_delta_threshold: float = 0.03, plateau_window: int = 3) -> FeedbackState:
    ordered_nodes = sorted(tree.nodes.values(), key=_node_order_key)
    attempts = [item for node in ordered_nodes if (item := _attempt_from_node(tree, node)) is not None]
    total_attempts = len(attempts)
    compile_success = sum(1 for item in attempts if item.compile_ok)
    correct_success = sum(1 for item in attempts if item.correct)
    best_node = max(
        (node for node in tree.nodes.values() if node.correct and node.speedup is not None),
        key=lambda node: node.speedup or float("-inf"),
        default=tree.get_node(tree.root_id),
    )
    best_speedup = best_node.speedup
    deltas = [
        (item.speedup or 0.0) - (item.parent_speedup or 0.0)
        for item in attempts
        if item.correct and item.speedup is not None and item.parent_speedup is not None
    ]
    positive = [delta for delta in deltas if delta > 0]
    negative = [delta for delta in deltas if delta <= 0]
    recent_lineage_attempts = [
        item for item in attempts
        if item.correct and item.parent_id in _lineage(tree, best_node.node_id)
    ][-plateau_window:]
    recent_positive_mutations = []
    recent_negative_mutations = []
    for item in attempts[-6:]:
        record = {
            "attempt_id": item.attempt_id,
            "strategy_name": item.strategy_name,
            "mode": item.mode,
            "mutation_family": item.mutation_family,
            "speedup": item.speedup,
            "parent_speedup": item.parent_speedup,
            "single_change_focus": item.single_change_focus,
        }
        if item.correct and item.speedup is not None and item.parent_speedup is not None and item.speedup > item.parent_speedup:
            recent_positive_mutations.append(record)
        elif item.speedup is not None and item.parent_speedup is not None:
            recent_negative_mutations.append(record)
    plateau_detected = (
        len(recent_lineage_attempts) >= plateau_window
        and all(
            item.speedup is not None and item.parent_speedup is not None and abs(item.speedup - item.parent_speedup) < plateau_delta_threshold
            for item in recent_lineage_attempts
        )
    )
    last_outcome = None
    if attempts:
        last = attempts[-1]
        if last.failure_type:
            last_outcome = "failure"
        elif last.speedup is not None and last.parent_speedup is not None and last.speedup > last.parent_speedup:
            last_outcome = "improved"
        else:
            last_outcome = "regressed_or_flat"
    champion = ChampionState(
        node_id=best_node.node_id,
        speedup=best_node.speedup,
        strategy_name=best_node.plan_strategy_name,
        mutation_family=best_node.mutation_family,
        lineage=_lineage(tree, best_node.node_id),
        recent_positive_mutations=recent_positive_mutations[-4:],
        recent_negative_mutations=recent_negative_mutations[-4:],
        plateau_detected=plateau_detected,
        lineage_plateau_depth=len(recent_lineage_attempts) if plateau_detected else 0,
    )
    phase = "refinement" if best_speedup is not None and best_speedup > 1.0 else "exploration"
    return FeedbackState(
        total_attempts=total_attempts,
        compile_rate=(compile_success / total_attempts) if total_attempts else 0.0,
        correct_rate=(correct_success / total_attempts) if total_attempts else 0.0,
        best_speedup=best_speedup,
        best_strategy_name=best_node.plan_strategy_name,
        phase=phase,
        current_champion_id=best_node.node_id,
        current_champion_speedup=best_speedup,
        plateau_detected=plateau_detected,
        recent_improvement_deltas=positive[-5:],
        recent_regression_deltas=negative[-5:],
        recent_successful_mutation_families=[
            str(item["mutation_family"]) for item in recent_positive_mutations if item.get("mutation_family")
        ][-5:],
        recent_failed_mutation_families=[
            str(item["mutation_family"]) for item in recent_negative_mutations if item.get("mutation_family")
        ][-5:],
        last_mutation_outcome=last_outcome,
        recent_attempts=attempts[-8:],
        champion=champion,
    )
