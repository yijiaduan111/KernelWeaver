import unittest
from pathlib import Path
import shutil
import tempfile
import importlib.util
import re

from stark.core.bridge import KernelBenchTaskBridge, selected_kernelbench_targets
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


class KernelbenchBridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = KernelBenchTaskBridge()
        self._tmp_root = Path(__file__).resolve().parents[1] / "runs"
        self._tmp_root.mkdir(parents=True, exist_ok=True)

    def _new_tmp_dir(self) -> Path:
        base = self._tmp_root / f"kw_bridge_{next(tempfile._get_candidate_names())}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def test_main_l1_15_tasks_are_curated(self):
        task_file = Path(__file__).resolve().parents[1] / "configs" / "tasks" / "main_l1_15.yaml"
        task_ids = {
            int(match.group(1))
            for match in re.finditer(r"^\s*problem_id:\s*(\d+)\s*$", task_file.read_text(encoding="utf-8"), re.MULTILINE)
        }
        curated_ids = {
            int(row["problem_id"])
            for row in selected_kernelbench_targets()
            if int(row["level"]) == 1
        }
        self.assertTrue(task_ids.issubset(curated_ids))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for bridge tests")
    def test_auto_bridge_loads_unselected_problem_from_direct_root(self):
        tmp = self._new_tmp_dir()
        try:
            root = tmp
            level_dir = root / "level1"
            level_dir.mkdir(parents=True, exist_ok=True)
            problem_path = level_dir / "2_AutoBridgeMatmul.py"
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
                        "    def forward(self, a, b):",
                        "        return torch.matmul(a, b)",
                        "",
                        "def get_inputs():",
                        "    return [torch.rand(1024, 1024), torch.rand(1024, 1024)]",
                        "",
                        "def get_init_inputs():",
                        "    return []",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            task = self.bridge.load_official_problem(root, 1, 2, backend="triton")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

        self.assertEqual(task.entry_kind, "model_class")
        self.assertEqual(task.backend, "triton")
        self.assertIn("auto_bridge", task.tags)
        self.assertEqual(len(task.test_cases), 1)
        self.assertEqual(len(task.benchmark_cases), 1)
        self.assertEqual(len(task.test_cases[0].args), 2)
        self.assertIn("class ModelNew", task.source_code)
        self.assertIn("# <<<IMPROVE:forward_body>>>", task.source_code)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for bridge tests")
    def test_auto_bridge_builds_generic_cuda_scaffold(self):
        tmp = self._new_tmp_dir()
        try:
            root = tmp
            level_dir = root / "KernelBench" / "level1"
            level_dir.mkdir(parents=True, exist_ok=True)
            problem_path = level_dir / "11_AutoBridgeSigmoid.py"
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
                        "        return torch.sigmoid(x)",
                        "",
                        "def get_inputs():",
                        "    return [torch.rand(4096, 4096)]",
                        "",
                        "def get_init_inputs():",
                        "    return []",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            task = self.bridge.load_official_problem(root, 1, 11, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

        self.assertEqual(task.backend, "cuda")
        self.assertIn("native_cuda", task.tags)
        self.assertIn("CUDA_CPP_SRC", task.source_code)
        self.assertIn("# <<<IMPROVE:cuda_cu>>>", task.source_code)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for bridge tests")
    def test_auto_bridge_builds_tilelang_scaffold(self):
        tmp = self._new_tmp_dir()
        try:
            root = tmp
            level_dir = root / "KernelBench" / "level1"
            level_dir.mkdir(parents=True, exist_ok=True)
            problem_path = level_dir / "13_AutoBridgeRelu.py"
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
                        "        return torch.relu(x)",
                        "",
                        "def get_inputs():",
                        "    return [torch.rand(2048, 2048)]",
                        "",
                        "def get_init_inputs():",
                        "    return []",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            task = self.bridge.load_official_problem(root, 1, 13, backend="tilelang")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

        self.assertEqual(task.backend, "tilelang")
        self.assertIn("tilelang", task.tags)
        self.assertIn("_stark_import_tilelang", task.source_code)
        self.assertIn("# <<<IMPROVE:tilelang_kernel>>>", task.source_code)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for bridge tests")
    def test_auto_bridge_builds_cute_scaffold(self):
        tmp = self._new_tmp_dir()
        try:
            root = tmp
            level_dir = root / "KernelBench" / "level1"
            level_dir.mkdir(parents=True, exist_ok=True)
            problem_path = level_dir / "14_AutoBridgeGelu.py"
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
                        "        return torch.nn.functional.gelu(x)",
                        "",
                        "def get_inputs():",
                        "    return [torch.rand(1024, 1024)]",
                        "",
                        "def get_init_inputs():",
                        "    return []",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            task = self.bridge.load_official_problem(root, 1, 14, backend="cute")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

        self.assertEqual(task.backend, "cute")
        self.assertIn("cute", task.tags)
        self.assertIn("_stark_import_cute", task.source_code)
        self.assertIn("# <<<IMPROVE:cute_kernel>>>", task.source_code)
