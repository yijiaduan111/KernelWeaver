import importlib.util
import re
import shutil
import tempfile
import types
import unittest
from pathlib import Path

from stark.core.bridge import KernelBenchTaskBridge, selected_kernelbench_targets
from stark.core.bridge_specs import SELECTED_TARGETS
from stark.core.workflow import run_stark
from stark.demo import build_demo_tasks
from stark.evaluation import DemoEvaluator
from stark.io import load_run, save_run
from stark.models import StarkConfig
from stark.providers import MockProvider


MAIN_L1_15_PROBLEMS = [1, 10, 20, 25, 33, 40, 42, 45, 47, 50, 61, 82, 89, 95, 97]


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

    @staticmethod
    def _kernelbench_root() -> Path:
        return Path(__file__).resolve().parents[1] / "KernelBench"

    @staticmethod
    def _clone_runtime_value(torch, value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, list):
            return [KernelbenchBridgeTests._clone_runtime_value(torch, item) for item in value]
        if isinstance(value, tuple):
            return tuple(KernelbenchBridgeTests._clone_runtime_value(torch, item) for item in value)
        if isinstance(value, dict):
            return {key: KernelbenchBridgeTests._clone_runtime_value(torch, item) for key, item in value.items()}
        return value

    @staticmethod
    def _load_module_from_source(name: str, source: str):
        module = types.ModuleType(name)
        exec(source, module.__dict__)
        return module

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

    def test_main_l1_15_targets_define_manual_forward_steps(self):
        for problem_id in MAIN_L1_15_PROBLEMS:
            target = SELECTED_TARGETS[(1, problem_id)]
            self.assertTrue(target.get("forward_steps"), f"L1/P{problem_id} is missing manual forward_steps")
            self.assertGreaterEqual(len(target["forward_steps"]), 3)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for bridge tests")
    def test_main_l1_15_triton_strategies_align_with_grounded_regions(self):
        root = self._kernelbench_root()
        for problem_id in MAIN_L1_15_PROBLEMS:
            task = self.bridge.load_official_problem(root, 1, problem_id, backend="triton")
            anchors = {region.anchor_name for region in task.grounded_regions}
            self.assertIn("forward_step_1", anchors, f"L1/P{problem_id} did not load manual forward step anchors")
            self.assertNotIn("forward_body", anchors, f"L1/P{problem_id} unexpectedly fell back to a single forward_body anchor")
            for strategy in task.strategy_catalog:
                self.assertIn(
                    strategy.anchor_name,
                    anchors,
                    f"L1/P{problem_id} strategy {strategy.name} targets missing anchor {strategy.anchor_name}",
                )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for bridge tests")
    def test_main_l1_15_triton_root_scaffolds_match_reference(self):
        import torch

        root = self._kernelbench_root()
        for problem_id in MAIN_L1_15_PROBLEMS:
            task = self.bridge.load_official_problem(root, 1, problem_id, backend="triton")
            case = task.test_cases[0]
            reference_module = self._load_module_from_source(f"reference_{problem_id}", task.reference_code)
            candidate_module = self._load_module_from_source(f"candidate_{problem_id}", task.source_code)

            reference_init_args = [self._clone_runtime_value(torch, value) for value in case.init_args]
            candidate_init_args = [self._clone_runtime_value(torch, value) for value in case.init_args]
            reference_inputs = [self._clone_runtime_value(torch, value) for value in case.args]
            candidate_inputs = [self._clone_runtime_value(torch, value) for value in case.args]

            torch.manual_seed(0)
            reference_model = reference_module.Model(*reference_init_args)
            torch.manual_seed(0)
            candidate_model = candidate_module.ModelNew(*candidate_init_args)

            with torch.no_grad():
                reference_output = reference_model(*reference_inputs)
                candidate_output = candidate_model(*candidate_inputs)

            self.assertTrue(
                torch.allclose(reference_output, candidate_output, atol=1e-4, rtol=1e-4),
                f"L1/P{problem_id} manual scaffold no longer matches the reference root behavior",
            )

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
    def test_curated_p45_uses_safe_cuda_scaffold(self):
        root = self._kernelbench_root()
        task = self.bridge.load_official_problem(root, 1, 45, backend="cuda")

        self.assertEqual(task.backend, "cuda")
        self.assertIn("cuda_safe_forward_only", task.tags)
        self.assertNotIn("native_cuda", task.tags)
        self.assertNotIn("CUDA_CPP_SRC", task.source_code)
        self.assertIn("# <<<IMPROVE:helpers>>>", task.source_code)
        self.assertIn("# <<<IMPROVE:init_body>>>", task.source_code)
        self.assertIn("# <<<IMPROVE:forward_step_1>>>", task.source_code)
        self.assertIn("# <<<IMPROVE:forward_step_2>>>", task.source_code)
        self.assertNotIn("# <<<IMPROVE:forward_body>>>", task.source_code)

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
        self.assertIn("cuda_safe_forward_only", task.tags)
        self.assertNotIn("native_cuda", task.tags)
        self.assertNotIn("CUDA_CPP_SRC", task.source_code)
        self.assertIn("# <<<IMPROVE:forward_body>>>", task.source_code)
