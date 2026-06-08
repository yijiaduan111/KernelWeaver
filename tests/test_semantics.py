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

    def _write_problem(
        self,
        root: Path,
        problem_id: int,
        name: str,
        init_signature: str,
        init_body: list[str],
        body: str,
        imports: list[str] | None = None,
        globals_: list[str] | None = None,
        get_inputs: list[str] | None = None,
        get_init_inputs: list[str] | None = None,
    ) -> None:
        level_dir = root / "KernelBench" / "level1"
        level_dir.mkdir(parents=True, exist_ok=True)
        import_lines = imports or ["import torch", "import torch.nn as nn"]
        global_lines = globals_ or []
        get_inputs_lines = get_inputs or ["return [torch.rand(16, 16)]"]
        get_init_lines = get_init_inputs or ["return []"]
        content = "\n".join(
            [
                *import_lines,
                *global_lines,
                "",
                "class Model(nn.Module):",
                f"    def __init__({init_signature}):",
                "        super().__init__()",
                *[f"        {line}" for line in init_body],
                "",
                "    def forward(self, x):",
                f"        {body}",
                "",
                "def get_inputs():",
                *[f"    {line}" for line in get_inputs_lines],
                "",
                "def get_init_inputs():",
                *[f"    {line}" for line in get_init_lines],
                "",
            ]
        )
        (level_dir / f"{problem_id}_{name}.py").write_text(content, encoding="utf-8")

    def test_swish_maps_to_elementwise_forward_anchor(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 25, "Swish", "self", [], "return x * torch.sigmoid(x)")
            task = self.loader.load_official_problem(tmp, 1, 25, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        profile = task.semantic_profile
        self.assertIsNotNone(profile)
        self.assertEqual(profile.op_type, "elementwise")
        self.assertIn("forward_stmt_1", profile.recommended_anchors)
        self.assertTrue(any(intent.name == "fuse_elementwise_ops" for intent in profile.optimization_intents))
        self.assertIsNone(profile.exact_facts)

    def test_sum_reduction_records_exact_axis_facts(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(
                tmp,
                47,
                "SumReduction",
                "self, dim",
                ["self.dim = dim"],
                "return torch.sum(x, dim=self.dim, keepdim=True)",
                globals_=["batch_size = 128", "dim1 = 4096", "dim2 = 4095", "reduce_dim = 1"],
                get_inputs=["return [torch.rand(batch_size, dim1, dim2)]"],
                get_init_inputs=["return [reduce_dim]"],
            )
            task = self.loader.load_official_problem(tmp, 1, 47, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        profile = task.semantic_profile
        self.assertIsNotNone(profile)
        self.assertEqual(profile.op_type, "reduction")
        self.assertIsNotNone(profile.exact_facts)
        self.assertEqual(profile.exact_facts.kind, "reduction_exact_axis")
        self.assertEqual(profile.exact_facts.details["reduce_dim"], 1)
        self.assertEqual(profile.exact_facts.details["reduced_extent"], 4096)
        self.assertTrue(profile.exact_facts.details["keepdim"])

    def test_layernorm_records_exact_normalized_axes(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(
                tmp,
                40,
                "LayerNorm",
                "self, normalized_shape",
                ["self.ln = nn.LayerNorm(normalized_shape=normalized_shape)"],
                "return self.ln(x)",
                globals_=["batch_size = 16", "features = 64", "dim1 = 256", "dim2 = 256"],
                get_inputs=["return [torch.rand(batch_size, features, dim1, dim2)]"],
                get_init_inputs=["return [(features, dim1, dim2)]"],
            )
            task = self.loader.load_official_problem(tmp, 1, 40, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        profile = task.semantic_profile
        self.assertIsNotNone(profile)
        self.assertEqual(profile.op_type, "normalization")
        self.assertIsNotNone(profile.exact_facts)
        self.assertEqual(profile.exact_facts.kind, "layernorm_exact_axes")
        self.assertEqual(profile.exact_facts.details["outer_extent"], 16)
        self.assertEqual(profile.exact_facts.details["inner_extent"], 64 * 256 * 256)
        self.assertEqual(profile.exact_facts.details["normalized_shape"], [64, 256, 256])

    def test_batchnorm_records_exact_channel_axis(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(
                tmp,
                33,
                "BatchNorm",
                "self, num_features",
                ["self.bn = nn.BatchNorm2d(num_features=num_features)"],
                "return self.bn(x)",
                globals_=["batch_size = 64", "features = 64", "dim1 = 512", "dim2 = 512"],
                get_inputs=["return [torch.rand(batch_size, features, dim1, dim2)]"],
                get_init_inputs=["return [features]"],
            )
            task = self.loader.load_official_problem(tmp, 1, 33, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        profile = task.semantic_profile
        self.assertIsNotNone(profile)
        self.assertEqual(profile.op_type, "normalization")
        self.assertIsNotNone(profile.exact_facts)
        self.assertEqual(profile.exact_facts.kind, "batchnorm_channel_axes")
        self.assertEqual(profile.exact_facts.details["channel_dim"], 1)
        self.assertEqual(profile.exact_facts.details["channel_extent"], 64)

    def test_unknown_pattern_keeps_loader_usable(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_problem(tmp, 99, "Unknown", "self", [], "return x")
            task = self.loader.load_official_problem(tmp, 1, 99, backend="cuda")
        finally:
            shutil.rmtree(tmp)
        self.assertEqual(task.semantic_profile.op_type, "unknown")
        self.assertIn("class ModelNew", task.source_code)
        self.assertIsNone(task.semantic_profile.exact_facts)


if __name__ == "__main__":
    unittest.main()
