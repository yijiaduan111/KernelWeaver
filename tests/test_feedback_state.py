import shutil
import unittest
from pathlib import Path

from stark.core.workflow import _attempt_mode_for_selected_node, run_stark
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


class WorkflowSelectionModeTests(unittest.TestCase):
    def test_root_selection_stays_explore(self):
        code = "# <<<IMPROVE:cuda_cu>>>\npass\n# <<<END_IMPROVE>>>\n"
        config = StarkConfig(max_attempts=10)
        root = SearchNode(node_id="root", parent_id=None, depth=0, code=code, origin="root", compile_ok=True, correct=True, node_status="correct")
        tree = TreeMemory(root, config)
        mode = _attempt_mode_for_selected_node(tree, "root", "exploit_best_score", FeedbackState())
        self.assertEqual(mode, "explore")

    def test_non_root_selection_stays_explore_while_root_is_champion(self):
        code = "# <<<IMPROVE:cuda_cu>>>\npass\n# <<<END_IMPROVE>>>\n"
        config = StarkConfig(max_attempts=10)
        root = SearchNode(node_id="root", parent_id=None, depth=0, code=code, origin="root", compile_ok=True, correct=True, node_status="correct", speedup=0.99)
        child = SearchNode(node_id="n1", parent_id="root", depth=1, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=0.85)
        root.child_ids = ["n1"]
        tree = TreeMemory(root, config)
        tree.nodes = {"root": root, "n1": child}
        feedback = FeedbackState(
            current_champion_id="root",
            current_champion_speedup=0.99,
            champion=ChampionState(node_id="root", lineage=["root"]),
        )
        mode = _attempt_mode_for_selected_node(tree, "n1", "explore_leaf", feedback)
        self.assertEqual(mode, "explore")

    def test_non_root_selection_stays_explore_when_champion_is_not_above_baseline(self):
        code = "# <<<IMPROVE:cuda_cu>>>\npass\n# <<<END_IMPROVE>>>\n"
        config = StarkConfig(max_attempts=10)
        root = SearchNode(node_id="root", parent_id=None, depth=0, code=code, origin="root", compile_ok=True, correct=True, node_status="correct", speedup=1.0)
        champion = SearchNode(node_id="n1", parent_id="root", depth=1, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=0.97)
        child = SearchNode(node_id="n2", parent_id="n1", depth=2, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=0.95)
        root.child_ids = ["n1"]
        champion.child_ids = ["n2"]
        tree = TreeMemory(root, config)
        tree.nodes = {"root": root, "n1": champion, "n2": child}
        feedback = FeedbackState(
            current_champion_id="n1",
            current_champion_speedup=0.97,
            champion=ChampionState(node_id="n1", lineage=["root", "n1"]),
        )
        mode = _attempt_mode_for_selected_node(tree, "n2", "explore_leaf", feedback)
        self.assertEqual(mode, "explore")

    def test_champion_node_with_children_maps_to_best_lineage_push(self):
        code = "# <<<IMPROVE:cuda_cu>>>\npass\n# <<<END_IMPROVE>>>\n"
        config = StarkConfig(max_attempts=10)
        root = SearchNode(node_id="root", parent_id=None, depth=0, code=code, origin="root", compile_ok=True, correct=True, node_status="correct")
        champion = SearchNode(node_id="n5", parent_id="root", depth=1, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=4.8)
        child = SearchNode(node_id="n7", parent_id="n5", depth=2, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=4.7)
        root.child_ids = ["n5"]
        champion.child_ids = ["n7"]
        tree = TreeMemory(root, config)
        tree.nodes = {"root": root, "n5": champion, "n7": child}
        feedback = FeedbackState(current_champion_id="n5", champion=ChampionState(node_id="n5", lineage=["root", "n5"]))
        mode = _attempt_mode_for_selected_node(tree, "n5", "exploit_best_score", feedback)
        self.assertEqual(mode, "best_lineage_push")
        descendant_mode = _attempt_mode_for_selected_node(tree, "n7", "explore_leaf", feedback)
        self.assertEqual(descendant_mode, "mutate_champion")

    def test_non_champion_branch_maps_to_challenger(self):
        code = "# <<<IMPROVE:cuda_cu>>>\npass\n# <<<END_IMPROVE>>>\n"
        config = StarkConfig(max_attempts=10)
        root = SearchNode(node_id="root", parent_id=None, depth=0, code=code, origin="root", compile_ok=True, correct=True, node_status="correct")
        champion = SearchNode(node_id="n5", parent_id="root", depth=1, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=4.8)
        sibling = SearchNode(node_id="n6", parent_id="root", depth=1, code=code, origin="plan_code", compile_ok=True, correct=True, node_status="correct", speedup=4.2)
        root.child_ids = ["n5", "n6"]
        tree = TreeMemory(root, config)
        tree.nodes = {"root": root, "n5": champion, "n6": sibling}
        feedback = FeedbackState(current_champion_id="n5", champion=ChampionState(node_id="n5", lineage=["root", "n5"]))
        mode = _attempt_mode_for_selected_node(tree, "n6", "explore_leaf", feedback)
        self.assertEqual(mode, "challenger")
