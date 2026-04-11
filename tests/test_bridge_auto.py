import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

from stark.core.bridge import KernelBenchTaskBridge


class AutoBridgeTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.mkdtemp(prefix="kernelweaver_bridge_")
        self.root = Path(self._temp_dir) / "KernelBench"
        (self.root / "level1").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _write_problem(self, name: str, body: str) -> Path:
        path = self.root / "level1" / name
        path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
        return path

    def test_auto_bridge_loads_non_curated_problem(self):
        self._write_problem(
            "2_AddOne.py",
            """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self, features=4):
                    super().__init__()
                    self.bias = nn.Parameter(torch.ones(features))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x + self.bias

            def get_inputs():
                return [torch.randn(2, 4)]

            def get_init_inputs():
                return [4]
            """,
        )
        task = KernelBenchTaskBridge().load_official_problem(Path(self._temp_dir), 1, 2, backend="triton")
        self.assertTrue(task.name.startswith("kernelbench_l1_2_"))
        self.assertIn("auto_bridge", task.tags)
        self.assertEqual(task.test_cases, [])
        self.assertEqual(task.benchmark_cases, [])
        self.assertIn("paper evaluator", task.description)
        self.assertIn("class ModelNew", task.source_code)
        self.assertIn("# <<<IMPROVE:forward_body>>>", task.source_code)

    def test_auto_bridge_builds_generic_cuda_scaffold(self):
        self._write_problem(
            "3_SimpleNorm.py",
            """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return torch.nn.functional.layer_norm(x, x.shape[1:])

            def get_inputs():
                return [torch.randn(2, 4, 4)]

            def get_init_inputs():
                return []
            """,
        )
        task = KernelBenchTaskBridge().load_official_problem(Path(self._temp_dir), 1, 3, backend="cuda")
        self.assertIn("auto_bridge", task.tags)
        self.assertIn("native_cuda", task.tags)
        self.assertIn('CUDA_CPP_SRC = r"""', task.source_code)
        self.assertIn('CUDA_CU_SRC = r"""', task.source_code)
        self.assertIn("class ModelNew", task.source_code)

    def test_curated_override_still_wins_when_available(self):
        self._write_problem(
            "25_Swish.py",
            """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x * torch.sigmoid(x)

            def get_inputs():
                return [torch.randn(2, 8)]

            def get_init_inputs():
                return []
            """,
        )
        task = KernelBenchTaskBridge().load_official_problem(Path(self._temp_dir), 1, 25, backend="triton")
        self.assertEqual(task.name, "kernelbench_l1_25_swish")
        self.assertNotIn("auto_bridge", task.tags)
        self.assertGreater(len(task.test_cases), 0)
        self.assertGreater(len(task.benchmark_cases), 0)
        self.assertGreater(len(task.strategy_catalog), 0)


if __name__ == "__main__":
    unittest.main()
