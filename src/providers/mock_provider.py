"""Deterministic provider used for tests and smoke runs."""

from __future__ import annotations

from ..models import AgentContext, AnchorEdit, PlanProposal, SearchNode, TaskSpec
from ..utils import apply_anchor_edit
from .base_provider import AgentProvider


class MockProvider(AgentProvider):
    """Deterministic provider used for tests and smoke runs."""

    name = "mock"

    def __init__(self) -> None:
        self._broken_once: set[tuple[str, str]] = set()

    def propose_plan(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> PlanProposal:
        del node
        strategies = task.strategy_catalog
        tried_here = {snapshot.plan_strategy_name for snapshot in context.related if snapshot.plan_strategy_name}
        leader_strategies = {snapshot.plan_strategy_name for snapshot in context.leaders if snapshot.plan_strategy_name}
        for strategy in strategies:
            if strategy.name not in tried_here and strategy.name not in leader_strategies:
                return PlanProposal(
                    strategy_name=strategy.name,
                    strategy_summary=strategy.strategy_summary,
                    anchor_edits=[
                        AnchorEdit(
                            anchor_name=strategy.anchor_name,
                            instruction=strategy.instruction,
                            operation="replace",
                        )
                    ],
                    expected_gain=strategy.expected_gain,
                    risk_notes="Use the existing grounded anchor and keep the edit local.",
                )
        strategy = strategies[0]
        return PlanProposal(
            strategy_name=strategy.name,
            strategy_summary=strategy.strategy_summary,
            anchor_edits=[
                AnchorEdit(
                    anchor_name=strategy.anchor_name,
                    instruction=strategy.instruction,
                    operation="replace",
                )
            ],
            expected_gain=strategy.expected_gain,
            risk_notes="Fallback to the first known grounded strategy.",
        )

    def generate_code(
        self,
        task: TaskSpec,
        node: SearchNode,
        proposal: PlanProposal,
        context: AgentContext,
    ) -> str:
        del context
        strategy = task.strategy_map()[proposal.strategy_name]
        edit = proposal.anchor_edits[0]
        broken_key = (task.name, strategy.name)
        body = strategy.good_body
        if strategy.broken_body is not None and broken_key not in self._broken_once:
            body = strategy.broken_body
            self._broken_once.add(broken_key)
        return apply_anchor_edit(node.code, edit.anchor_name, body, operation=edit.operation)

    def debug_code(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> str:
        del context
        if node.plan_strategy_name is None:
            raise ValueError("Cannot debug a node without a plan strategy.")
        strategy = task.strategy_map()[node.plan_strategy_name]
        edit = node.anchor_edits[0] if node.anchor_edits else AnchorEdit(anchor_name=strategy.anchor_name, instruction="repair", operation="replace")
        body = strategy.debug_body or strategy.good_body
        return apply_anchor_edit(node.code, edit.anchor_name, body, operation=edit.operation)
