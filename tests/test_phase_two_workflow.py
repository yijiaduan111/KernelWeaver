import json
import unittest

from stark.core.workflow import run_stark
from stark.deliberation.runner import MultiModelDeliberationRunner
from stark.models import AgentContext, AnchorEdit, EvaluationResult, PlanProposal, SearchNode, StarkConfig, TaskSpec
from stark.providers import MockProvider
from stark.providers.base_provider import AgentProvider
from stark.providers.openai_provider import _task_metadata


class DeterministicPhaseProvider(AgentProvider):
    def propose_plan(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> PlanProposal:
        del task
        return PlanProposal(
            strategy_name="phase_strategy",
            strategy_summary="Local sum rewrite",
            anchor_edits=[AnchorEdit(anchor_name="body", instruction="Replace body with the next deterministic variant.")],
            expected_gain="Deterministic phase-two test candidate.",
            target_node_id=node.node_id,
            target_anchors=["body"],
            attempt_mode=context.attempt_mode or "explore",
        )

    def generate_code(
        self,
        task: TaskSpec,
        node: SearchNode,
        proposal: PlanProposal,
        context: AgentContext,
    ) -> str:
        del task, proposal, context
        if "return sum(data)" in node.code:
            body = "return sum(data) + 0"
        else:
            body = "return sum(data)"
        return json.dumps({"region_patches": [{"region": "body", "operation": "replace", "body": body}]})

    def debug_code(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> str:
        del task, context
        return node.code


class DeterministicEvaluator:
    def evaluate(self, task: TaskSpec, candidate_code: str, config: StarkConfig) -> EvaluationResult:
        del task, config
        if "return sum(data) + 0" in candidate_code:
            runtime = 0.4
            speedup = 2.5
        elif "return sum(data)" in candidate_code:
            runtime = 0.5
            speedup = 2.0
        else:
            runtime = 1.0
            speedup = 1.0
        return EvaluationResult(
            compile_ok=True,
            correct=True,
            runtime=runtime,
            score=runtime,
            logs=[],
            failure_type=None,
            failure_stage="none",
            reference_runtime=1.0,
            speedup=speedup,
            reference_runtimes={"torch_eager": 1.0},
            speedups={"torch_eager": speedup},
            primary_reference="torch_eager",
        )


def _task() -> TaskSpec:
    source = (
        "class ModelNew:\n"
        "    def forward(self, data):\n"
        "        # <<<IMPROVE:body>>>\n"
        "        total = 0\n"
        "        for item in data:\n"
        "            total += item\n"
        "        return total\n"
        "        # <<<END_IMPROVE>>>\n"
    )
    return TaskSpec(
        name="phase-two-workflow-task",
        description="Deterministic phase-two workflow smoke task.",
        source_code=source,
        reference_code=source,
        function_name="forward",
        reference_function_name="forward",
        test_cases=[],
        benchmark_cases=[],
    )


class PhaseTwoWorkflowTests(unittest.TestCase):
    def test_phase_two_reroot_creates_phase_transition(self):
        task = _task()
        config = StarkConfig(
            max_attempts=4,
            phase_two_enabled=True,
            phase_two_split_attempts=2,
            deliberation_enabled=True,
            deliberation_profile="quick",
            run_profile="quick",
            search_profile="quick",
            evaluator_profile="quick",
            measurement_profile="quick",
        )
        deliberation_runner = MultiModelDeliberationRunner(
            providers={"mock": MockProvider()},
            max_strategies=2,
            strategies_per_model=1,
        )
        task.strategy_portfolio = deliberation_runner.run(task, config)
        result = run_stark(
            task,
            config,
            DeterministicPhaseProvider(),
            DeterministicEvaluator(),
            deliberation_runner=deliberation_runner,
        )

        self.assertIsNotNone(result.phase_transition)
        self.assertEqual(result.phase_transition.target_phase, 2)
        self.assertTrue(any(node_id.startswith("phase2_") for node_id in result.nodes))
        self.assertTrue(any(node.origin == "phase2_root" for node in result.nodes.values()))
        metadata = _task_metadata(task)
        self.assertIsNotNone(metadata.get("phase_transition"))
