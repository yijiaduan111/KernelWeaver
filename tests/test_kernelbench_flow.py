import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from stark.core.loader import KernelBenchLoader
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
            run_profile="quick",
            search_profile="quick",
            evaluator_profile="quick",
            measurement_profile="quick",
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


class KernelbenchLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = KernelBenchLoader()
        self._tmp_root = Path(__file__).resolve().parents[1] / "runs"
        self._tmp_root.mkdir(parents=True, exist_ok=True)

    def _new_tmp_dir(self) -> Path:
        base = self._tmp_root / f"kw_loader_{next(tempfile._get_candidate_names())}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _write_problem(self, root: Path, problem_id: int, body: str) -> Path:
        level_dir = root / "KernelBench" / "level1"
        level_dir.mkdir(parents=True, exist_ok=True)
        problem_path = level_dir / f"{problem_id}_SyntheticProblem.py"
        problem_path.write_text(
            "\n".join(
                [
                    "import torch",
                    "import torch.nn as nn",
                    "",
                    "class Model(nn.Module):",
                    "    def __init__(self):",
                    "        super().__init__()",
                    "",
                    "    def forward(self, x):",
                    f"        {body}",
                    "",
                    "def get_inputs():",
                    "    return [torch.rand(16, 16)]",
                    "",
                    "def get_init_inputs():",
                    "    return []",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return problem_path

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for loader tests")
    def test_loader_builds_raw_triton_scaffold(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 2, "return torch.relu(x)")
            task = self.loader.load_official_problem(tmp, 1, 2, backend="triton")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

        self.assertEqual(task.entry_kind, "model_class")
        self.assertEqual(task.backend, "triton")
        self.assertEqual(task.strategy_catalog, [])
        self.assertEqual(task.test_cases, [])
        self.assertEqual(task.benchmark_cases, [])
        self.assertIn("class ModelNew", task.source_code)
        self.assertIn("# <<<IMPROVE:forward_body>>>", task.source_code)
        self.assertIn("return torch.relu(x)", task.source_code)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for loader tests")
    def test_loader_builds_cuda_scaffold(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 11, "return torch.sigmoid(x)")
            task = self.loader.load_official_problem(tmp, 1, 11, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

        self.assertEqual(task.backend, "cuda")
        self.assertIn("CUDA_CPP_SRC", task.source_code)
        self.assertIn("# <<<IMPROVE:cuda_cu>>>", task.source_code)
        self.assertIn("return torch.sigmoid(x)", task.source_code)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for loader tests")
    def test_loader_builds_tilelang_and_cute_scaffolds(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 13, "return x + 1")
            tilelang_task = self.loader.load_official_problem(tmp, 1, 13, backend="tilelang")
            cute_task = self.loader.load_official_problem(tmp, 1, 13, backend="cute")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

        self.assertIn("# <<<IMPROVE:tilelang_kernel>>>", tilelang_task.source_code)
        self.assertIn("# <<<IMPROVE:cute_kernel>>>", cute_task.source_code)
        self.assertEqual(tilelang_task.strategy_catalog, [])
        self.assertEqual(cute_task.strategy_catalog, [])
