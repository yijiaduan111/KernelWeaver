import unittest
from pathlib import Path
import shutil

from stark.core.workflow import run_stark
from stark.demo import build_demo_tasks
from stark.evaluation import DemoEvaluator
from stark.io import load_run, save_run
from stark.models import StarkConfig
from stark.providers import MockProvider


class KernelbenchFlowTests(unittest.TestCase):
    def test_demo_flow_can_save_and_reload(self):
        task = build_demo_tasks()[0]
        config = StarkConfig(
            max_attempts=2,
            benchmark_loops=1,
            warmup_loops=0,
            run_profile='quick',
            search_profile='quick',
            evaluator_profile='quick',
            measurement_profile='quick',
        )
        result = run_stark(task, config, MockProvider(), DemoEvaluator())
        tmpdir = Path(__file__).resolve().parents[1] / "runs" / ".tmp_test_flow"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            run_path = save_run(result, tmpdir)
            reloaded = load_run(run_path)
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)
        self.assertEqual(reloaded.task_name, task.name)
        self.assertIn(reloaded.best_node_id, reloaded.nodes)
