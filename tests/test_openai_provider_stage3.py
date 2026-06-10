import json
import unittest
from unittest.mock import patch

from stark.diagnostics.schema import MachineCheckProfile, TaskDiagnostics
from stark.deliberation.schema import DeliberationStrategy, StrategyPortfolio
from stark.memory.schema import MemoryMethodCard, MemoryProfile
from stark.models import AgentContext, AnchorEdit, NodeSnapshot, PlanProposal, SearchNode, TaskSpec
from stark.providers import OpenAICompatibleConfig, OpenAICompatibleProvider
from stark.providers.openai_provider import _task_metadata


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
        plan_mode="explore",
        mutation_family="baseline",
        single_change_focus="none",
    )


class OpenAIProviderStage3Tests(unittest.TestCase):
    def test_task_metadata_includes_machine_check_profile(self):
        task = _task()
        task.diagnostics_profile = TaskDiagnostics(
            enabled=True,
            mode="machine_check_v1",
            machine_check_profile=MachineCheckProfile(
                enabled=True,
                status="ok",
                case_id="CASE_1",
                allowed_methods=["Launch_Tuning"],
                forbidden_methods=["RegisterBlocking"],
            ),
        )
        metadata = _task_metadata(task)
        self.assertIn("diagnostics_profile", metadata)
        self.assertIn("machine_check_profile", metadata)
        self.assertEqual(metadata["machine_check_profile"]["allowed_methods"], ["Launch_Tuning"])

    def test_propose_plan_falls_back_to_selected_strategy_memory_family(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="x"))
        task = _task()
        task.diagnostics_profile = TaskDiagnostics(
            enabled=True,
            mode="machine_check_v1",
            machine_check_profile=MachineCheckProfile(
                enabled=True,
                status="ok",
                allowed_methods=["SharedMemoryTiling"],
            ),
        )
        task.memory_profile = MemoryProfile(
            enabled=True,
            bootstrap_cards=[
                MemoryMethodCard(
                    method_id="SharedMemoryTiling",
                    title="Shared Memory Tiling",
                    summary="row-wise tiling",
                )
            ],
        )
        task.strategy_portfolio = StrategyPortfolio(
            enabled=True,
            strategies=[
                DeliberationStrategy(
                    strategy_id="strategy_01",
                    intent="tiling",
                    summary="rowwise tiling",
                    memory_methods=["SharedMemoryTiling"],
                )
            ],
        )
        node = SearchNode(node_id="n1", parent_id="root", depth=1, code=task.source_code, origin="plan_code")
        context = AgentContext(
            role="plan",
            current=_snapshot("n1", 1.7),
            root=_snapshot("root", 1.0),
            attempt_mode="explore",
        )
        response = {
            "strategy_name": "strategy_01",
            "strategy_summary": "Refine cuda kernel",
            "expected_gain": "small",
            "risk_notes": "low",
            "mode": "explore",
            "target_node_id": "n1",
            "target_anchors": ["cuda_cu"],
            "frozen_anchors": [],
            "change_budget": "medium",
            "must_preserve": [],
            "reason_against_rewrite": "",
            "performance_hypothesis": "tile rows",
            "single_change_focus": "tile rows",
            "target_metric": "speedup",
            "anchor_edits": [{"anchor_name": "cuda_cu", "instruction": "tile rows", "operation": "replace"}],
        }
        with patch.object(provider, "_chat", return_value=json.dumps(response)):
            proposal = provider.propose_plan(task, node, context)
        self.assertEqual(proposal.mutation_family, "SharedMemoryTiling")

    def test_propose_plan_parses_stage3_fields(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="x"))
        task = _task()
        task.diagnostics_profile = TaskDiagnostics(
            enabled=True,
            mode="machine_check_v1",
            machine_check_profile=MachineCheckProfile(
                enabled=True,
                status="ok",
                allowed_methods=["Launch_Tuning"],
                forbidden_methods=["RegisterBlocking"],
            ),
        )
        node = SearchNode(node_id="n1", parent_id="root", depth=1, code=task.source_code, origin="plan_code")
        context = AgentContext(
            role="plan",
            current=_snapshot("n1", 1.7),
            root=_snapshot("root", 1.0),
            attempt_mode="mutate_champion",
            best_node=_snapshot("n0", 2.0),
            best_kernel_summary={"speedup": 2.0},
            best_kernel_code=task.source_code,
            active_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
            champion=_snapshot("n0", 2.0),
            champion_summary={"node_id": "n0", "speedup": 2.0},
            recent_positive_mutations=[{"mutation_family": "family_a"}],
            recent_negative_mutations=[{"mutation_family": "family_b"}],
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
            "performance_hypothesis": "memory access dominates",
            "single_change_focus": "tune block size",
            "mutation_family": "launch_tuning",
            "target_metric": "speedup",
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
        self.assertEqual(proposal.attempt_mode, "mutate_champion")
        self.assertEqual(proposal.single_change_focus, "tune block size")
        self.assertEqual(proposal.mutation_family, "launch_tuning")

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
            attempt_mode="mutate_champion",
            target_node_id="n1",
            target_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
            change_budget="small",
            performance_hypothesis="memory access dominates",
            single_change_focus="tune block size",
            mutation_family="launch_tuning",
        )
        context = AgentContext(
            role="code",
            current=_snapshot("n1", 1.7),
            root=_snapshot("root", 1.0),
            attempt_mode="mutate_champion",
            active_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
            best_kernel_excerpt={"cuda_cu": "return x"},
            best_kernel_summary={"speedup": 2.0},
            champion=_snapshot("n0", 2.0),
            champion_code=task.source_code,
            champion_summary={"node_id": "n0", "speedup": 2.0},
        )
        with patch.object(
            provider,
            "_chat",
            return_value='{"region_patches":[{"region":"cuda_cu","operation":"replace","body":"return x"}]}',
        ):
            out = provider.generate_code(task, node, proposal, context)
        payload = json.loads(out)
        self.assertEqual(payload["region_patches"][0]["region"], "cuda_cu")


    def test_propose_plan_mutation_prompt_uses_excerpt_not_full_code(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="x"))
        task = _task()
        task.diagnostics_profile = TaskDiagnostics(
            enabled=True,
            mode="machine_check_v1",
            machine_check_profile=MachineCheckProfile(
                enabled=True,
                status="ok",
                allowed_methods=["Launch_Tuning"],
            ),
        )
        node = SearchNode(node_id="n1", parent_id="root", depth=1, code=task.source_code, origin="plan_code")
        context = AgentContext(
            role="plan",
            current=_snapshot("n1", 1.7),
            root=_snapshot("root", 1.0),
            attempt_mode="mutate_champion",
            best_node=_snapshot("n0", 2.0),
            best_kernel_summary={"speedup": 2.0},
            best_kernel_code=task.source_code,
            best_kernel_excerpt={"cuda_cu": "return x"},
            active_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
            champion=_snapshot("n0", 2.0),
            champion_summary={"node_id": "n0", "speedup": 2.0},
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
            "performance_hypothesis": "memory access dominates",
            "single_change_focus": "tune block size",
            "mutation_family": "launch_tuning",
            "target_metric": "speedup",
            "anchor_edits": [{"anchor_name": "cuda_cu", "instruction": "tune block size", "operation": "replace"}],
        }
        def fake_chat(*, system_prompt, user_payload, temperature, reasoning_effort):
            self.assertIn("allowed_methods", system_prompt)
            self.assertEqual(user_payload["task_metadata"]["machine_check_profile"]["allowed_methods"], ["Launch_Tuning"])
            self.assertIn("If attempt_mode is mutate_champion", system_prompt)
            self.assertIn("single_change_focus", system_prompt)
            self.assertNotIn("best_kernel_code", user_payload)
            self.assertEqual(user_payload["best_kernel_excerpt"], {"cuda_cu": "return x"})
            return json.dumps(response)
        with patch.object(provider, "_chat", side_effect=fake_chat):
            proposal = provider.propose_plan(task, node, context)
        self.assertEqual(proposal.attempt_mode, "mutate_champion")
        self.assertEqual(proposal.single_change_focus, "tune block size")
        self.assertEqual(proposal.mutation_family, "launch_tuning")

    def test_generate_code_prompt_enforces_single_change_focus(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="x"))
        task = _task()
        node = SearchNode(node_id="n1", parent_id="root", depth=1, code=task.source_code, origin="plan_code")
        proposal = PlanProposal(
            strategy_name="strategy_01",
            strategy_summary="Refine",
            anchor_edits=[AnchorEdit(anchor_name="cuda_cu", instruction="refine", operation="replace")],
            expected_gain="small",
            mode="refine",
            attempt_mode="mutate_champion",
            target_node_id="n1",
            target_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
            change_budget="small",
            performance_hypothesis="memory access dominates",
            single_change_focus="tune block size",
            mutation_family="launch_tuning",
        )
        context = AgentContext(
            role="code",
            current=_snapshot("n1", 1.7),
            root=_snapshot("root", 1.0),
            attempt_mode="mutate_champion",
            active_anchors=["cuda_cu"],
            frozen_anchors=["user_helpers"],
            best_kernel_excerpt={"cuda_cu": "return x"},
            best_kernel_summary={"speedup": 2.0},
            champion=_snapshot("n0", 2.0),
            champion_code=task.source_code,
            champion_summary={"node_id": "n0", "speedup": 2.0},
        )
        def fake_chat(*, system_prompt, user_payload, temperature, reasoning_effort):
            self.assertIn("plan.single_change_focus", system_prompt)
            self.assertIn("exactly that one local change", system_prompt)
            self.assertEqual(user_payload["plan"]["single_change_focus"], "tune block size")
            return '{"region_patches":[{"region":"cuda_cu","operation":"replace","body":"return x"}]}'
        with patch.object(provider, "_chat", side_effect=fake_chat):
            out = provider.generate_code(task, node, proposal, context)
        payload = json.loads(out)
        self.assertEqual(payload["region_patches"][0]["region"], "cuda_cu")


if __name__ == "__main__":
    unittest.main()
