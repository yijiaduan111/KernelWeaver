import shutil
import unittest
from pathlib import Path

from stark.core.workflow import run_stark
from stark.demo import build_demo_tasks
from stark.evaluation import DemoEvaluator
from stark.io import load_run, save_run
from stark.models import StarkConfig
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
