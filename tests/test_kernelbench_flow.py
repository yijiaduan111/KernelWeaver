import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from stark.core.loader import KernelBenchLoader
from stark.core.workflow import run_stark
from stark.demo import build_demo_tasks
from stark.evaluation import DemoEvaluator
from stark.io import load_run, save_run
from stark.models import StarkConfig
from stark.providers import MockProvider
from src.utils import normalize_anchor_scaffold
from src.evaluation.evaluator_paper import (
    _paper_eval_timeout_seconds,
    _paper_exception_stage,
    _paper_failure_from_metadata,
)


class KernelbenchFlowTests(unittest.TestCase):
    def test_paper_timeout_uses_config_with_bounded_environment_override(self):
        config = StarkConfig(paper_eval_timeout_seconds=240)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STARK_PAPER_EVAL_TIMEOUT_SECONDS", None)
            self.assertEqual(_paper_eval_timeout_seconds(config), 240)
        with patch.dict(os.environ, {"STARK_PAPER_EVAL_TIMEOUT_SECONDS": "120"}):
            self.assertEqual(_paper_eval_timeout_seconds(config), 120)
        with patch.dict(os.environ, {"STARK_PAPER_EVAL_TIMEOUT_SECONDS": "10"}):
            self.assertEqual(_paper_eval_timeout_seconds(config), 60)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("dotenv") is not None,
        "torch and python-dotenv are required",
    )
    def test_kernelbench_input_processing_preserves_index_dtypes(self):
        import torch
        from kernelbench.eval import _process_input_tensor

        labels = torch.tensor([0, 2, 1], dtype=torch.int64)
        mask = torch.tensor([True, False, True], dtype=torch.bool)
        activations = torch.ones(3, dtype=torch.float64)

        self.assertEqual(_process_input_tensor(labels, torch.device("cpu")).dtype, torch.int64)
        self.assertEqual(_process_input_tensor(mask, torch.device("cpu")).dtype, torch.bool)
        self.assertEqual(
            _process_input_tensor(activations, torch.device("cpu"), precision=torch.float32).dtype,
            torch.float32,
        )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required")
    def test_paper_exception_classifies_nvcc_failure_before_generic_cuda_text(self):
        import torch

        stage, failure_type = _paper_exception_stage(
            torch,
            RuntimeError("Error building extension 'x': FAILED: cuda.cuda.o; ptxas error"),
        )
        self.assertEqual((stage, failure_type), ("compile", "cuda_compile_error"))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required")
    def test_paper_exception_keeps_kernel_launch_failure_as_runtime(self):
        import torch

        stage, _failure_type = _paper_exception_stage(
            torch,
            RuntimeError("CUDA kernel launch error: invalid configuration argument"),
        )
        self.assertEqual(stage, "runtime")

    def test_paper_metadata_reclassifies_lazy_extension_build_as_compile(self):
        stage, failure_type = _paper_failure_from_metadata(
            {
                "runtime_error_name": "builtins.RuntimeError",
                "runtime_error": "Error building extension 'x': FAILED: cuda.cuda.o",
                "runtime_error_traceback": "ptxas error; ninja: build stopped",
            }
        )
        self.assertEqual((stage, failure_type), ("compile", "cuda_compile_error"))

    def test_anchor_normalization_keeps_only_candidate_anchor_bodies(self):
        original = "header\n# <<<IMPROVE:x>>>\nold\n# <<<END_IMPROVE>>>\nfooter\n"
        candidate = "changed header\n# <<<IMPROVE:x>>>\nnew\n# <<<END_IMPROVE>>>\nchanged footer\n"
        self.assertEqual(
            normalize_anchor_scaffold(original, candidate),
            "header\n# <<<IMPROVE:x>>>\nnew\n# <<<END_IMPROVE>>>\nfooter\n",
        )

    def test_anchor_normalization_rejects_missing_or_reordered_markers(self):
        original = (
            "# <<<IMPROVE:a>>>\na\n# <<<END_IMPROVE>>>\n"
            "# <<<IMPROVE:b>>>\nb\n# <<<END_IMPROVE>>>\n"
        )
        reordered = (
            "# <<<IMPROVE:b>>>\nb\n# <<<END_IMPROVE>>>\n"
            "# <<<IMPROVE:a>>>\na\n# <<<END_IMPROVE>>>\n"
        )
        self.assertIsNone(normalize_anchor_scaffold(original, reordered))

    def test_stark_preserves_partial_run_on_llm_provider_error(self):
        class FailingProvider:
            def propose_plan(self, task, node, context):
                raise RuntimeError("LLM response content is empty")

        task = build_demo_tasks()[0]
        config = StarkConfig(max_attempts=1)
        result = run_stark(task, config, FailingProvider(), DemoEvaluator())
        self.assertEqual(result.stats["provider_errors"], 1)
        self.assertEqual(set(result.nodes), {"root"})

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
        self.assertIn("# <<<IMPROVE:forward_stmt_1>>>", task.source_code)
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
