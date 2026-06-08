import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stark.core.execution_facts import ExecutionDerived, ExecutionFacts, TensorFact, infer_workload_profile
from stark.core.loader import KernelBenchLoader
from stark.deliberation.runner import _proposal_payload
from stark.providers.openai_provider import _task_metadata
from stark.semantics import semantic_profile_to_prompt_dict


class ExecutionFactsInferenceTests(unittest.TestCase):
    def test_reduction_workload_tag_is_deterministic(self):
        facts = ExecutionFacts(
            input_tensors=[TensorFact(name="input_0", shape=[32, 2048], dtype="float32", device="cpu", numel=65536)],
            derived=ExecutionDerived(
                primary_shape=[32, 2048],
                tensor_count=1,
                has_broadcast_inputs=False,
                outer_size=32,
                inner_size=2048,
            ),
        )
        workload_tag, bottleneck_hint = infer_workload_profile("reduction", facts)
        self.assertEqual(workload_tag, "few_rows_medium_inner")
        self.assertEqual(bottleneck_hint, "unknown")

    def test_elementwise_broadcast_is_detected(self):
        facts = ExecutionFacts(
            input_tensors=[
                TensorFact(name="input_0", shape=[16, 16], dtype="float32", device="cpu", numel=256),
                TensorFact(name="input_1", shape=[16, 1], dtype="float32", device="cpu", numel=16),
            ],
            derived=ExecutionDerived(
                primary_shape=[16, 16],
                tensor_count=2,
                has_broadcast_inputs=True,
                outer_size=16,
                inner_size=16,
            ),
        )
        workload_tag, bottleneck_hint = infer_workload_profile("elementwise", facts)
        self.assertEqual(workload_tag, "elementwise_broadcast")
        self.assertEqual(bottleneck_hint, "unknown")

    def test_reduction_degrades_to_unknown_when_shape_facts_are_invalid(self):
        facts = ExecutionFacts(
            derived=ExecutionDerived(
                primary_shape=[],
                tensor_count=0,
                has_broadcast_inputs=False,
                outer_size=0,
                inner_size=2048,
            ),
        )
        workload_tag, bottleneck_hint = infer_workload_profile("reduction", facts)
        self.assertEqual(workload_tag, "unknown")
        self.assertEqual(bottleneck_hint, "unknown")


class ExecutionFactsLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = KernelBenchLoader()
        self.tmp_root = Path(__file__).resolve().parents[1] / "runs"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def _new_tmp_dir(self) -> Path:
        path = self.tmp_root / f"kw_execfacts_{next(tempfile._get_candidate_names())}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_problem(self, root: Path, problem_id: int, body: str, get_inputs_body: list[str]) -> None:
        level_dir = root / "KernelBench" / "level1"
        level_dir.mkdir(parents=True, exist_ok=True)
        (level_dir / f"{problem_id}_ExecutionFacts.py").write_text(
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
                    *[f"    {line}" for line in get_inputs_body],
                    "",
                    "def get_init_inputs():",
                    "    return []",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for loader tests")
    def test_loader_extracts_execution_facts(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 31, "return x.sum(dim=-1)", ["return [torch.rand(32, 2048)]"])
            task = self.loader.load_official_problem(tmp, 1, 31, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        self.assertIsNotNone(task.execution_facts)
        self.assertEqual(task.execution_facts.derived.primary_shape, [32, 2048])
        self.assertEqual(task.semantic_profile.workload_tag, "few_rows_medium_inner")
        self.assertIn("workload_tag", semantic_profile_to_prompt_dict(task.semantic_profile))

    def test_loader_retries_probe_with_env_python_when_sys_executable_fails(self):
        payload = {
            "input_tensors": [{"name": "input_0", "shape": [32, 2048], "dtype": "float32", "device": "cpu", "numel": 65536}],
            "init_tensors": [],
            "init_args": [],
            "derived": {
                "primary_shape": [32, 2048],
                "tensor_count": 1,
                "has_broadcast_inputs": False,
                "outer_size": 32,
                "inner_size": 2048,
            },
        }
        attempted: list[str] = []

        def fake_run(command, **kwargs):
            del kwargs
            attempted.append(command[0])
            if command[0] == "bad-python":
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="ModuleNotFoundError: No module named 'torch'")
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        with patch.object(KernelBenchLoader, "_probe_python_candidates", return_value=["bad-python", "good-python"]):
            with patch("stark.core.loader.subprocess.run", side_effect=fake_run):
                facts = self.loader._extract_execution_facts(Path("synthetic_problem.py"))

        self.assertEqual(attempted, ["bad-python", "good-python"])
        self.assertIsNotNone(facts)
        self.assertEqual(facts.derived.primary_shape, [32, 2048])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for loader tests")
    def test_loader_preserves_non_tensor_init_args(self):
        tmp = self._new_tmp_dir()
        try:
            level_dir = tmp / "KernelBench" / "level1"
            level_dir.mkdir(parents=True, exist_ok=True)
            (level_dir / "40_LayerNorm.py").write_text(
                "\n".join(
                    [
                        "import torch",
                        "import torch.nn as nn",
                        "",
                        "class Model(nn.Module):",
                        "    def __init__(self, normalized_shape):",
                        "        super().__init__()",
                        "        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)",
                        "",
                        "    def forward(self, x):",
                        "        return self.ln(x)",
                        "",
                        "def get_inputs():",
                        "    return [torch.rand(16, 64, 256, 256)]",
                        "",
                        "def get_init_inputs():",
                        "    return [(64, 256, 256)]",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            task = self.loader.load_official_problem(tmp, 1, 40, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        self.assertIsNotNone(task.execution_facts)
        self.assertEqual(task.execution_facts.init_args, [[64, 256, 256]])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for loader tests")
    def test_loader_degrades_when_factories_fail(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 32, "return x + 1", ["raise RuntimeError('boom')"])
            task = self.loader.load_official_problem(tmp, 1, 32, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        self.assertIsNone(task.execution_facts)
        self.assertIsNotNone(task.semantic_profile)


class ExecutionFactsPromptTests(unittest.TestCase):
    def setUp(self):
        self.loader = KernelBenchLoader()
        self.tmp_root = Path(__file__).resolve().parents[1] / "runs"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def _new_tmp_dir(self) -> Path:
        path = self.tmp_root / f"kw_execfacts_prompt_{next(tempfile._get_candidate_names())}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_problem(self, root: Path) -> None:
        level_dir = root / "KernelBench" / "level1"
        level_dir.mkdir(parents=True, exist_ok=True)
        (level_dir / "47_SumReduction.py").write_text(
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
                    "        return torch.sum(x, dim=-1)",
                    "",
                    "def get_inputs():",
                    "    return [torch.rand(32, 2048)]",
                    "",
                    "def get_init_inputs():",
                    "    return []",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required for prompt tests")
    def test_prompt_payloads_include_execution_facts(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp)
            task = self.loader.load_official_problem(tmp, 1, 47, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        task_meta = _task_metadata(task)
        proposal = _proposal_payload(task, "mock", 2)
        self.assertIn("execution_facts", task_meta)
        self.assertEqual(task_meta["execution_facts"]["input_tensors"][0]["shape"], [32, 2048])
        self.assertEqual(task_meta["execution_facts"]["init_args"], [])
        self.assertIn("execution_facts", proposal)
        self.assertEqual(proposal["execution_facts"]["input_tensors"][0]["shape"], [32, 2048])
        self.assertEqual(proposal["execution_facts"]["init_args"], [])
