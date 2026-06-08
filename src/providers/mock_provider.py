"""Deterministic provider used for tests and smoke runs."""

from __future__ import annotations

import json

from ..models import AgentContext, AnchorEdit, PlanProposal, SearchNode, TaskSpec
from ..utils import extract_anchor_names
from .base_provider import AgentProvider


class MockProvider(AgentProvider):
    """Deterministic provider used for tests and smoke runs."""

    name = "mock"

    def __init__(self) -> None:
        self._broken_once: set[tuple[str, str]] = set()

    def propose_plan(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> PlanProposal:
        strategies = task.strategy_catalog
        tried_here = {snapshot.plan_strategy_name for snapshot in context.related if snapshot.plan_strategy_name}
        leader_strategies = {snapshot.plan_strategy_name for snapshot in context.leaders if snapshot.plan_strategy_name}
        for strategy in strategies:
            if strategy.name not in tried_here and strategy.name not in leader_strategies:
                return PlanProposal(
                    strategy_name=strategy.name,
                    strategy_summary=strategy.strategy_summary,
                    mode="explore",
                    target_node_id=node.node_id,
                    target_anchors=[strategy.anchor_name],
                    frozen_anchors=[],
                    change_budget="medium",
                    must_preserve=["Keep edits local to the chosen anchored region."],
                    reason_against_rewrite="Mock strategy should stay local.",
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
        anchors = extract_anchor_names(node.code)
        portfolio = getattr(task, "strategy_portfolio", None)
        if portfolio is not None and portfolio.strategies:
            used = tried_here | leader_strategies
            for strategy in portfolio.strategies:
                if strategy.strategy_id in used:
                    continue
                target = next((anchor for anchor in strategy.target_anchors if anchor in anchors), anchors[0] if anchors else "helpers")
                return PlanProposal(
                    strategy_name=strategy.strategy_id,
                    strategy_summary=strategy.summary,
                    mode="explore",
                    target_node_id=node.node_id,
                    target_anchors=[target],
                    frozen_anchors=[],
                    change_budget="medium",
                    must_preserve=["Keep edits local to the chosen grounded anchor."],
                    reason_against_rewrite="Mock portfolio strategy should stay local.",
                    anchor_edits=[
                        AnchorEdit(
                            anchor_name=target,
                            instruction="; ".join(strategy.implementation_hints) or strategy.summary,
                            operation="replace",
                        )
                    ],
                    expected_gain=strategy.expected_gain,
                    risk_notes="; ".join(strategy.risk_notes),
                )
        anchor_name = "forward_body" if "forward_body" in anchors else (anchors[0] if anchors else "helpers")
        return PlanProposal(
            strategy_name="mock_structural_plan",
            strategy_summary="Use the generic structural anchor selected by the loader.",
            anchor_edits=[AnchorEdit(anchor_name=anchor_name, instruction="Keep the baseline implementation unchanged.", operation="replace")],
            expected_gain="Smoke-test the loader and workflow without handwritten task strategies.",
            risk_notes="No task-specific strategy catalog is available.",
            mode="explore",
            target_node_id=node.node_id,
            target_anchors=[anchor_name],
            frozen_anchors=[],
            change_budget="medium",
            must_preserve=["Keep the scaffold unchanged outside the requested region."],
            reason_against_rewrite="Mock fallback should stay structural and local.",
        )

    def generate_code(
        self,
        task: TaskSpec,
        node: SearchNode,
        proposal: PlanProposal,
        context: AgentContext,
    ) -> str:
        del context
        edit = proposal.anchor_edits[0]
        strategy = task.strategy_map().get(proposal.strategy_name)
        if strategy is None:
            return json.dumps({"region_patches": [{"region": edit.anchor_name, "operation": edit.operation, "body": _region_body(node.code, edit.anchor_name)}]})
        broken_key = (task.name, strategy.name)
        body = strategy.good_body
        if strategy.broken_body is not None and broken_key not in self._broken_once:
            body = strategy.broken_body
            self._broken_once.add(broken_key)
        return json.dumps({"region_patches": [{"region": edit.anchor_name, "operation": edit.operation, "body": body}]})

    def debug_code(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> str:
        del context
        if node.plan_strategy_name is None:
            raise ValueError("Cannot debug a node without a plan strategy.")
        strategy = task.strategy_map().get(node.plan_strategy_name)
        if strategy is None:
            anchors = extract_anchor_names(node.code)
            fallback_anchor = node.anchor_edits[0] if node.anchor_edits else AnchorEdit(anchor_name=(anchors[0] if anchors else "helpers"), instruction="repair", operation="replace")
            return json.dumps({"region_patches": [{"region": fallback_anchor.anchor_name, "operation": fallback_anchor.operation, "body": _region_body(node.code, fallback_anchor.anchor_name)}]})
        edit = node.anchor_edits[0] if node.anchor_edits else AnchorEdit(anchor_name=strategy.anchor_name, instruction="repair", operation="replace")
        body = strategy.debug_body or strategy.good_body
        return json.dumps({"region_patches": [{"region": edit.anchor_name, "operation": edit.operation, "body": body}]})

    def generate_text(
        self,
        system_prompt: str,
        user_payload: dict,
        temperature: float = 0.2,
        purpose: str = "generic",
    ) -> str:
        del system_prompt, temperature
        provider_name = str(user_payload.get("provider_name") or self.name)
        if purpose == "deliberation_review":
            portfolio = user_payload.get("strategy_portfolio") or {}
            scores = [
                {"strategy_id": item.get("strategy_id"), "score": 4, "notes": f"{provider_name} mock review"}
                for item in portfolio.get("strategies", [])
                if item.get("strategy_id")
            ]
            return json.dumps({"scores": scores})
        anchors = list(user_payload.get("available_anchors") or [])
        target = next((anchor for anchor in anchors if str(anchor).startswith("forward_stmt_")), anchors[0] if anchors else "helpers")
        return json.dumps(
            {
                "strategies": [
                    {
                        "intent": f"{provider_name}_mock_fusion",
                        "summary": f"Use {target} for a local mock optimization plan.",
                        "target_anchors": [target],
                        "implementation_hints": ["Keep the public task interface unchanged.", "Edit only the selected grounded anchor."],
                        "expected_gain": "Exercise deliberation wiring in tests.",
                        "risk_notes": ["Mock strategy does not optimize real performance."],
                        "score": 4,
                    }
                ]
            }
        )


def _region_body(source_code: str, region: str) -> str:
    import re
    import textwrap
    pattern = re.compile(r"(?ms)^[ \t]*#\s*<<<IMPROVE:" + re.escape(region) + r">>>(?:\r?\n)(?P<body>.*?)(^[ \t]*#\s*<<<END_IMPROVE>>>)")
    match = pattern.search(source_code)
    if not match:
        return "pass"
    return textwrap.dedent(match.group("body")).strip("\n")
