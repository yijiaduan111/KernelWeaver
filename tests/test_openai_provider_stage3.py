import json
import unittest
from unittest.mock import patch

from stark.models import AgentContext, AnchorEdit, NodeSnapshot, PlanProposal, SearchNode, TaskSpec
from stark.providers import OpenAICompatibleConfig, OpenAICompatibleProvider


def _task() -> TaskSpec:
    source = """
class ModelNew:
    def forward(self, x):
        # <<<IMPROVE:cuda_cu>>>
        return x
        # <<<END_IMPROVE>>>
"""
    return TaskSpec(
        name="stage3-task",
        description="",
        source_code=source,
        reference_code=source,
        function_name="forward",
        reference_function_name="forward",
        test_cases=[],
        benchmark_cases=[],
        benchmark_family="kernelbench",
        backend="cuda",
    )


def _snapshot(node_id: str, speedup: float | None = None) -> NodeSnapshot:
    return NodeSnapshot(
        node_id=node_id,
        parent_id=None,
        depth=0,
        score=1.0,
        status="correct",
        plan_strategy_name="strategy_01",
        failure_type=None,
        child_count=0,
        origin="plan_code",
        selected_count=0,
        runtime=1.0,
        latest_failure_stage=None,
        speedup=speedup,
    )


class OpenAIProviderStage3Tests(unittest.TestCase):
    def test_propose_plan_parses_stage3_fields(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="x"))
        task = _task()
        node = SearchNode(node_id="n1", parent_id="root", depth=1, code=task.source_code, origin="plan_code")
        context = AgentContext(
            role="plan",
            current=_snapshot("n1", 1.7),
            root=_snapshot("root", 1.0),
            best_node=_snapshot("n0", 2.0),
            best_kernel_summary={"speedup": 2.0},
            best_kernel_code=task.source_code,
            active_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
        )
        response = {
            "strategy_name": "strategy_01",
            "strategy_summary": "Refine cuda kernel",
            "expected_gain": "small",
            "risk_notes": "low",
            "mode": "refine",
            "target_node_id": "n0",
            "target_anchors": ["cuda_cu"],
            "frozen_anchors": ["user_helpers"],
            "change_budget": "small",
            "must_preserve": ["keep launch structure"],
            "reason_against_rewrite": "preserve working kernel",
            "anchor_edits": [
                {"anchor_name": "cuda_cu", "instruction": "tune block size", "operation": "replace"}
            ],
        }
        with patch.object(provider, "_chat", return_value=json.dumps(response)):
            proposal = provider.propose_plan(task, node, context)
        self.assertEqual(proposal.mode, "refine")
        self.assertEqual(proposal.target_node_id, "n0")
        self.assertEqual(proposal.target_anchors, ["cuda_cu"])
        self.assertEqual(proposal.frozen_anchors, ["user_helpers"])
        self.assertEqual(proposal.change_budget, "small")
        self.assertEqual(proposal.must_preserve, ["keep launch structure"])

    def test_generate_code_normalizes_requested_regions_only(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="x"))
        task = _task()
        node = SearchNode(node_id="n1", parent_id="root", depth=1, code=task.source_code, origin="plan_code")
        proposal = PlanProposal(
            strategy_name="strategy_01",
            strategy_summary="Refine",
            anchor_edits=[AnchorEdit(anchor_name="cuda_cu", instruction="refine", operation="replace")],
            expected_gain="small",
            mode="refine",
            target_node_id="n1",
            target_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
            change_budget="small",
        )
        context = AgentContext(
            role="code",
            current=_snapshot("n1", 1.7),
            root=_snapshot("root", 1.0),
            active_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
            best_kernel_excerpt={"cuda_cu": "return x"},
            best_kernel_summary={"speedup": 2.0},
        )
        with patch.object(
            provider,
            "_chat",
            return_value='{"region_patches":[{"region":"cuda_cu","operation":"replace","body":"return x"}]}',
        ):
            out = provider.generate_code(task, node, proposal, context)
        payload = json.loads(out)
        self.assertEqual(payload["region_patches"][0]["region"], "cuda_cu")


if __name__ == "__main__":
    unittest.main()
