import shutil
import unittest
from pathlib import Path

from stark.core.workflow import _attempt_mode_for_index, _select_node_for_mode, run_stark
from stark.demo import build_demo_tasks
from stark.core.tree import TreeMemory
from stark.evaluation import DemoEvaluator
from stark.io import load_run, save_run
from stark.feedback.schema import ChampionState, FeedbackState
from stark.models import SearchNode, StarkConfig
from stark.providers import MockProvider


class FeedbackStateTests(unittest.TestCase):
    def test_run_persists_feedback_state(self):
        task = build_demo_tasks()[0]
        config = StarkConfig(
            max_attempts=2,
            benchmark_loops=1,
            warmup_loops=0,
            run_profile="quick",
            search_profile="quick",
            evaluator_profile="quick",
            measurement_profile="quick",
        )
        result = run_stark(task, config, MockProvider(), DemoEvaluator())
        self.assertIsNotNone(result.feedback_state)
        self.assertGreaterEqual(result.feedback_state.total_attempts, 1)
        self.assertIn(result.feedback_state.phase, {"exploration", "exploitation", "refinement"})

        tmpdir = Path(__file__).resolve().parents[1] / "runs" / ".tmp_test_feedback_state"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            run_path = save_run(result, tmpdir)
            reloaded = load_run(run_path)
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

        self.assertIsNotNone(reloaded.feedback_state)
        self.assertEqual(reloaded.feedback_state.total_attempts, result.feedback_state.total_attempts)
        self.assertEqual(reloaded.feedback_state.phase, result.feedback_state.phase)
        self.assertEqual(reloaded.feedback_state.best_strategy_name, result.feedback_state.best_strategy_name)
        self.assertEqual(reloaded.feedback_state.current_champion_id, result.feedback_state.current_champion_id)
        self.assertEqual(reloaded.feedback_state.champion.node_id, result.feedback_state.champion.node_id)


class WorkflowSchedulingTests(unittest.TestCase):
    def test_attempt_schedule_reserves_three_challengers_and_final_push(self):
        config = StarkConfig(max_attempts=10, explore_fraction=0.4, challenger_fraction=0.3)
        modes = [_attempt_mode_for_index(i, config, None) for i in range(1, 11)]
        self.assertEqual(
            modes,
            ["explore", "explore", "explore", "explore", "mutate_champion", "mutate_champion", "challenger", "challenger", "challenger", "best_lineage_push"],
        )

    def test_plateau_keeps_two_mutation_attempts_before_switching(self):
        config = StarkConfig(max_attempts=10, explore_fraction=0.4, challenger_fraction=0.3, plateau_recovery_mutation_attempts=2)
        feedback = FeedbackState(plateau_detected=True)
        modes = [_attempt_mode_for_index(i, config, feedback) for i in range(5, 11)]
        self.assertEqual(modes, ["mutate_champion", "mutate_champion", "mutate_champion", "mutate_champion", "challenger", "best_lineage_push"])

    def test_best_lineage_push_prefers_frontier_descendant_over_champion(self):
        code = "# <<<IMPROVE:cuda_cu>>>\npass\n# <<<END_IMPROVE>>>\n"
        config = StarkConfig(max_attempts=10)
        root = SearchNode(node_id="root", parent_id=None, depth=0, code=code, origin="root", compile_ok=True, correct=True, node_status="correct")
        champion = SearchNode(node_id="n5", parent_id="root", depth=1, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=4.8)
        frontier = SearchNode(node_id="n7", parent_id="n5", depth=2, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=4.7)
        root.child_ids = ["n5"]
        champion.child_ids = ["n7"]
        tree = TreeMemory(root, config)
        tree.nodes = {"root": root, "n5": champion, "n7": frontier}
        feedback = FeedbackState(current_champion_id="n5", champion=ChampionState(node_id="n5", lineage=["root", "n5"]))
        selected = _select_node_for_mode(tree, config, "best_lineage_push", feedback)
        self.assertEqual(selected, ("n7", "best_lineage_push"))
