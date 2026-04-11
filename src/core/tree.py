from __future__ import annotations

import math
import random
from copy import deepcopy

from ..models import EvaluationResult, PlanProposal, SearchNode, StarkConfig


def _status_from_evaluation(evaluation: EvaluationResult) -> str:
    if evaluation.failure_stage == "compile":
        return "compile_fail"
    if evaluation.failure_stage == "runtime":
        return "runtime_fail"
    if evaluation.failure_stage == "correctness":
        return "correctness_fail"
    return "correct"


class TreeMemory:
    def __init__(self, root_node: SearchNode, config: StarkConfig) -> None:
        self.root_id = root_node.node_id
        self.nodes: dict[str, SearchNode] = {root_node.node_id: root_node}
        self.leaderboard: list[str] = []
        self.leaderboard_history: list[list[str]] = [[]]
        self.pruned_nodes: dict[str, str] = {}
        self.last_exclusions: dict[str, str] = {}
        self.selection_exclusion_history: list[dict[str, str]] = []
        self._counter = 0
        self._rng = random.Random(config.seed)

    def get_node(self, node_id: str) -> SearchNode:
        return self.nodes[node_id]

    def add_child(
        self,
        parent_id: str,
        code: str,
        proposal: PlanProposal,
        evaluation: EvaluationResult,
        origin: str,
    ) -> SearchNode:
        parent = self.nodes[parent_id]
        self._counter += 1
        node_id = f"n{self._counter}"
        child = SearchNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            code=code,
            origin=origin,
            plan_strategy_name=proposal.strategy_name,
            plan_summary=proposal.strategy_summary,
            anchor_edits=list(proposal.anchor_edits),
            compile_ok=evaluation.compile_ok,
            correct=evaluation.correct,
            runtime=evaluation.runtime,
            score=evaluation.score,
            logs=list(evaluation.logs),
            failure_type=evaluation.failure_type,
            node_status=_status_from_evaluation(evaluation),
            latest_failure_stage=None if evaluation.failure_stage == "none" else evaluation.failure_stage,
            reference_runtime=evaluation.reference_runtime,
            speedup=evaluation.speedup,
            reference_runtimes=dict(evaluation.reference_runtimes),
            speedups=dict(evaluation.speedups),
            primary_reference=evaluation.primary_reference,
        )
        self.nodes[node_id] = child
        parent.child_ids.append(node_id)
        return child

    def update_leaderboard(self, node_id: str, config: StarkConfig) -> None:
        node = self.nodes[node_id]
        if node.compile_ok and node.correct:
            if node_id not in self.leaderboard:
                self.leaderboard.append(node_id)
            self.leaderboard.sort(
                key=lambda current_id: (
                    self.nodes[current_id].score,
                    self.nodes[current_id].depth,
                    current_id,
                )
            )
            self.leaderboard = self.leaderboard[: config.leaderboard_size]
        self._record_leaderboard_history()

    def _record_leaderboard_history(self) -> None:
        snapshot = list(self.leaderboard)
        if not self.leaderboard_history or self.leaderboard_history[-1] != snapshot:
            self.leaderboard_history.append(snapshot)

    def subtree_ids(self, node_id: str) -> set[str]:
        pending = [node_id]
        subtree: set[str] = set()
        while pending:
            current = pending.pop()
            if current in subtree:
                continue
            subtree.add(current)
            pending.extend(self.nodes[current].child_ids)
        return subtree

    def sibling_ids(self, node_id: str) -> list[str]:
        node = self.nodes[node_id]
        if node.parent_id is None:
            return []
        parent = self.nodes[node.parent_id]
        return [child_id for child_id in parent.child_ids if child_id != node_id]

    def cousin_child_ids(self, node_id: str) -> list[str]:
        cousins: list[str] = []
        for sibling_id in self.sibling_ids(node_id):
            cousins.extend(self.nodes[sibling_id].child_ids)
        return cousins

    def refresh_pruned_nodes(self, config: StarkConfig) -> None:
        for node_id, node in self.nodes.items():
            if node_id in self.pruned_nodes:
                continue
            if len(node.child_ids) < config.dead_branch_threshold:
                continue
            if all(self.nodes[child_id].is_failure for child_id in node.child_ids):
                reason = "dead_branch_pruned"
                node.node_status = "pruned"
                node.prune_reason = reason
                self.pruned_nodes[node_id] = reason

    def is_pruned(self, node_id: str, config: StarkConfig) -> bool:
        self.refresh_pruned_nodes(config)
        return node_id in self.pruned_nodes

    def exclusion_reason(self, node_id: str, config: StarkConfig) -> str | None:
        self.refresh_pruned_nodes(config)
        node = self.nodes[node_id]
        if node_id in self.pruned_nodes:
            return self.pruned_nodes[node_id]
        if node_id == self.root_id and len(node.child_ids) >= config.root_child_limit:
            return "root_throttled"
        if node.is_failure and node.debug_attempts >= config.debug_retry_limit:
            return "debug_retry_limit_reached"
        return None

    def is_eligible(self, node_id: str, config: StarkConfig) -> bool:
        return self.exclusion_reason(node_id, config) is None

    def is_expandable(self, node_id: str, config: StarkConfig) -> bool:
        return self.is_eligible(node_id, config)

    def eligibility_report(self, config: StarkConfig) -> tuple[list[str], dict[str, str]]:
        self.refresh_pruned_nodes(config)
        eligible: list[str] = []
        excluded: dict[str, str] = {}
        for node_id in self.nodes:
            reason = self.exclusion_reason(node_id, config)
            if reason is None:
                eligible.append(node_id)
            else:
                excluded[node_id] = reason
        return eligible, excluded

    def eligible_nodes(self, config: StarkConfig) -> list[str]:
        eligible, _ = self.eligibility_report(config)
        return eligible

    def expandable_nodes(self, config: StarkConfig) -> list[str]:
        return self.eligible_nodes(config)

    def eligible_leaf_nodes(self, config: StarkConfig) -> list[str]:
        self.refresh_pruned_nodes(config)
        return [
            node_id
            for node_id, node in self.nodes.items()
            if self.is_eligible(node_id, config) and not node.child_ids
        ]

    def expandable_leaves(self, config: StarkConfig) -> list[str]:
        return self.eligible_leaf_nodes(config)

    def select_node(self, config: StarkConfig) -> tuple[str, str] | None:
        candidates, exclusions = self.eligibility_report(config)
        self.last_exclusions = dict(exclusions)
        self.selection_exclusion_history.append(dict(exclusions))
        if not candidates:
            return None
        leaves = self.eligible_leaf_nodes(config)
        explore = self._rng.random() < config.epsilon
        if explore and leaves:
            selected = self._rng.choice(leaves)
            reason = "debug_retry_failure" if self.nodes[selected].is_failure else "explore_leaf"
        else:
            finite = [node_id for node_id in candidates if math.isfinite(self.nodes[node_id].score)]
            if finite:
                selected = min(
                    finite,
                    key=lambda node_id: (
                        self.nodes[node_id].score,
                        self.nodes[node_id].depth,
                        self.nodes[node_id].selected_count,
                        node_id,
                    ),
                )
                reason = "debug_retry_failure" if self.nodes[selected].is_failure else "exploit_best_score"
            else:
                pool = leaves or candidates
                selected = min(
                    pool,
                    key=lambda node_id: (
                        self.nodes[node_id].selected_count,
                        self.nodes[node_id].depth,
                        node_id,
                    ),
                )
                reason = "debug_retry_failure" if self.nodes[selected].is_failure else "fallback_no_finite_score"
        self.nodes[selected].selected_count += 1
        self.nodes[selected].selection_reason = reason
        return selected, reason

    def clone_nodes(self) -> dict[str, SearchNode]:
        return deepcopy(self.nodes)
