import shutil
import tempfile
import unittest
from pathlib import Path

from stark.core.loader import KernelBenchLoader


class SemanticProfileTests(unittest.TestCase):
    def setUp(self):
        self.loader = KernelBenchLoader()
        self.tmp_root = Path(__file__).resolve().parents[1] / "runs"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def _new_tmp_dir(self) -> Path:
        path = self.tmp_root / f"kw_semantics_{next(tempfile._get_candidate_names())}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_problem(self, root: Path, problem_id: int, name: str, body: str, imports: list[str] | None = None) -> None:
        level_dir = root / "KernelBench" / "level1"
        level_dir.mkdir(parents=True, exist_ok=True)
        import_lines = imports or ["import torch", "import torch.nn as nn"]
        (level_dir / f"{problem_id}_{name}.py").write_text(
            "\n".join(
                [
                    *import_lines,
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

    def test_swish_maps_to_elementwise_forward_anchor(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 25, "Swish", "return x * torch.sigmoid(x)")
            task = self.loader.load_official_problem(tmp, 1, 25, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        profile = task.semantic_profile
        self.assertIsNotNone(profile)
        self.assertEqual(profile.op_type, "elementwise")
        self.assertIn("forward_stmt_1", profile.recommended_anchors)
        self.assertTrue(any(intent.name == "fuse_elementwise_ops" for intent in profile.optimization_intents))

    def test_sum_reduction_records_reduction_risk(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 47, "SumReduction", "return torch.sum(x, dim=1)")
            task = self.loader.load_official_problem(tmp, 1, 47, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        profile = task.semantic_profile
        self.assertIsNotNone(profile)
        self.assertEqual(profile.op_type, "reduction")
        self.assertTrue(any("reduction dimension" in note for note in profile.risk_notes))

    def test_layernorm_records_normalization_risk(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(
                tmp,
                40,
                "LayerNorm",
                "return F.layer_norm(x, x.shape[-1:])",
                imports=["import torch", "import torch.nn as nn", "import torch.nn.functional as F"],
            )
            task = self.loader.load_official_problem(tmp, 1, 40, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        profile = task.semantic_profile
        self.assertIsNotNone(profile)
        self.assertEqual(profile.op_type, "normalization")
        self.assertTrue(any("statistics" in note for note in profile.risk_notes))

    def test_unknown_pattern_keeps_loader_usable(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 99, "Unknown", "return x")
            task = self.loader.load_official_problem(tmp, 1, 99, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        self.assertEqual(task.semantic_profile.op_type, "unknown")
        self.assertIn("class ModelNew", task.source_code)
