import shutil
import tempfile
import unittest
from pathlib import Path

from stark.core.execution_facts import ExecutionFacts, TensorFact
from stark.core.loader import KernelBenchLoader
from stark.deliberation.runner import _proposal_payload
from stark.providers.openai_provider import _task_metadata
from stark.semantics import semantic_profile_from_dict, semantic_profile_to_prompt_dict


class ExecutionFactsSchemaTests(unittest.TestCase):
    def test_from_dict_preserves_only_raw_facts(self):
        payload = {
            "input_tensors": [{"name": "input_0", "shape": [32, 2048], "dtype": "float32", "numel": 65536}],
            "init_tensors": [],
            "init_args": [[1, 2, 3]],
            "derived": {"outer_size": 32, "inner_size": 2048},
        }
        facts = ExecutionFacts.from_dict(payload)
        assert facts is not None
        self.assertEqual(facts.input_tensors[0].shape, [32, 2048])
        self.assertEqual(facts.init_args, [[1, 2, 3]])
        self.assertNotIn("derived", facts.to_dict())

    def test_semantic_profile_ignores_legacy_heuristic_fields(self):
        profile = semantic_profile_from_dict(
            {
                "enabled": True,
                "mode": "rule",
                "op_type": "normalization",
                "summary": "legacy payload",
                "source": "legacy.py",
                "workload_tag": "many_rows_small_inner",
                "bottleneck_hint": "unknown",
                "recommended_anchors": ["forward_stmt_1"],
                "risk_notes": ["preserve semantics"],
            }
        )
        assert profile is not None
        payload = semantic_profile_to_prompt_dict(profile)
        assert payload is not None
        self.assertNotIn("workload_tag", payload)
        self.assertNotIn("bottleneck_hint", payload)

    def test_prompt_dict_contains_only_raw_fields(self):
        facts = ExecutionFacts(
            input_tensors=[
                TensorFact(name="input_0", shape=[16, 16], dtype="float32", numel=256),
                TensorFact(name="input_1", shape=[16, 1], dtype="float32", numel=16),
            ],
            init_args=[7, [8, 9]],
        )
        prompt = facts.to_prompt_dict()
        self.assertEqual(set(prompt.keys()), {"input_tensors", "init_tensors", "init_args"})
        self.assertNotIn("device", prompt["input_tensors"][0])
        self.assertEqual(prompt["input_tensors"][0]["shape"], [16, 16])
        self.assertEqual(prompt["init_args"], [7, [8, 9]])


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

    def _write_problem(
        self,
        root: Path,
        problem_id: int,
        body: str,
        get_inputs_body: list[str],
        init_body: list[str] | None = None,
    ) -> None:
        level_dir = root / "KernelBench" / "level1"
        level_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(
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
                *[f"    {line}" for line in (init_body or ["return []"])],
                "",
            ]
        )
        (level_dir / f"{problem_id}_ExecutionFacts.py").write_text(content, encoding="utf-8")

    def test_loader_extracts_raw_execution_facts_statically(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 31, "return x.sum(dim=-1)", ["return [torch.rand(32, 2048)]"])
            task = self.loader.load_official_problem(tmp, 1, 31, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        self.assertIsNotNone(task.execution_facts)
        self.assertEqual(task.execution_facts.input_tensors[0].shape, [32, 2048])
        self.assertEqual(task.execution_facts.input_tensors[0].dtype, "float32")

    def test_loader_preserves_non_tensor_init_args(self):
        tmp = self._new_tmp_dir()
        try:
            level_dir = tmp / "KernelBench" / "level1"
            level_dir.mkdir(parents=True, exist_ok=True)
            content = "\n".join(
                [
                    "import torch",
                    "import torch.nn as nn",
                    "batch_size = 16",
                    "features = 64",
                    "dim1 = 256",
                    "dim2 = 256",
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
                    "    return [torch.rand(batch_size, features, dim1, dim2)]",
                    "",
                    "def get_init_inputs():",
                    "    return [(features, dim1, dim2)]",
                    "",
                ]
            )
            (level_dir / "40_LayerNorm.py").write_text(content, encoding="utf-8")
            task = self.loader.load_official_problem(tmp, 1, 40, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        self.assertIsNotNone(task.execution_facts)
        self.assertEqual(task.execution_facts.init_args, [[64, 256, 256]])
        self.assertEqual(task.execution_facts.input_tensors[0].shape, [16, 64, 256, 256])

    def test_loader_extracts_randint_dtype(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 95, "return x", ["return [torch.rand(32, 128), torch.randint(0, 10, (32,))]"])
            task = self.loader.load_official_problem(tmp, 1, 95, backend="cuda")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        self.assertIsNotNone(task.execution_facts)
        self.assertEqual(task.execution_facts.input_tensors[1].dtype, "int64")
        self.assertEqual(task.execution_facts.input_tensors[1].shape, [32])

    def test_loader_degrades_when_static_parse_fails(self):
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
        content = "\n".join(
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
        )
        (level_dir / "47_SumReduction.py").write_text(content, encoding="utf-8")

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
        semantic_prompt = task_meta["semantic_profile"]
        self.assertIsNotNone(semantic_prompt)
        self.assertNotIn("workload_tag", semantic_prompt)
        self.assertNotIn("bottleneck_hint", semantic_prompt)


if __name__ == "__main__":
    unittest.main()
