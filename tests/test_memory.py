import shutil
import tempfile
import unittest
from pathlib import Path

from stark.core.loader import KernelBenchLoader
from stark.diagnostics.schema import MachineCheckProfile, TaskDiagnostics
from stark.deliberation.runner import _proposal_payload
from stark.deliberation.schema import DeliberationStrategy, StrategyPortfolio
from stark.feedback.schema import ChampionState, FeedbackState
from stark.memory import build_memory_profile, memory_profile_to_prompt_dict, refresh_memory_profile
from stark.providers.openai_provider import _task_metadata


class MemoryBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.loader = KernelBenchLoader()
        self.tmp_root = Path(__file__).resolve().parents[1] / "runs"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def _new_tmp_dir(self) -> Path:
        path = self.tmp_root / f"kw_memory_{next(tempfile._get_candidate_names())}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_layernorm_problem(self, root: Path) -> None:
        level_dir = root / "KernelBench" / "level1"
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

    def test_layernorm_bootstrap_uses_exact_semantic_facts(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_layernorm_problem(tmp)
            task = self.loader.load_official_problem(tmp, 1, 40, backend="cuda")
            task.diagnostics_profile = TaskDiagnostics(
                enabled=True,
                mode="machine_check_v1",
                machine_check_profile=MachineCheckProfile(
                    enabled=True,
                    status="ok",
                    tier="Tier-H",
                    bottleneck_id="shared_memory_capacity",
                    case_id="LN_SHARED_MEM",
                    allowed_methods=["SharedMemoryTiling", "Launch_Tuning"],
                    forbidden_methods=["RegisterBlocking"],
                    notes=["synthetic unit-test machine-check result"],
                ),
            )
            task.memory_profile = build_memory_profile(task, enabled=True, max_primary_cards=4, max_challenger_cards=3)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        self.assertIsNotNone(task.memory_profile)
        assert task.memory_profile is not None
        self.assertTrue(task.memory_profile.enabled)
        self.assertGreaterEqual(len(task.memory_profile.bootstrap_cards), 1)
        self.assertEqual(task.memory_profile.bootstrap_cards[0].method_id, "SharedMemoryTiling")
        prompt = memory_profile_to_prompt_dict(task.memory_profile)
        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertEqual(prompt["primary_methods"][0]["method_id"], "SharedMemoryTiling")
        self.assertIn("preferred_methods", prompt)
        self.assertIn("memory_profile", _task_metadata(task))
        self.assertIn("memory_profile", _proposal_payload(task, "mock", 2))

    def test_refresh_updates_feedback_digest_and_portfolio_order(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_layernorm_problem(tmp)
            task = self.loader.load_official_problem(tmp, 1, 40, backend="cuda")
            task.diagnostics_profile = TaskDiagnostics(
                enabled=True,
                mode="machine_check_v1",
                machine_check_profile=MachineCheckProfile(
                    enabled=True,
                    status="ok",
                    tier="Tier-H",
                    bottleneck_id="shared_memory_capacity",
                    case_id="LN_SHARED_MEM",
                    allowed_methods=["SharedMemoryTiling", "Launch_Tuning", "Reduce_Live_Ranges"],
                    forbidden_methods=[],
                    notes=["synthetic unit-test machine-check result"],
                ),
            )
            task.memory_profile = build_memory_profile(task, enabled=True, max_primary_cards=4, max_challenger_cards=3)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        assert task.memory_profile is not None
        task.strategy_portfolio = StrategyPortfolio(
            enabled=True,
            strategies=[
                DeliberationStrategy(
                    strategy_id="strategy_01",
                    intent="rowwise_tiling",
                    summary="rowwise tiled layernorm",
                    memory_methods=["SharedMemoryTiling"],
                    priority=4,
                ),
                DeliberationStrategy(
                    strategy_id="strategy_02",
                    intent="launch_tune",
                    summary="launch tuning challenger",
                    memory_methods=["Launch_Tuning"],
                    priority=3,
                ),
            ],
        )
        feedback = FeedbackState(
            total_attempts=4,
            compile_rate=1.0,
            correct_rate=1.0,
            best_speedup=4.2,
            current_champion_id="n4",
            plateau_detected=True,
            recent_successful_mutation_families=["SharedMemoryTiling"],
            recent_failed_mutation_families=["Reduce_Live_Ranges"],
            champion=ChampionState(
                node_id="n4",
                speedup=4.2,
                mutation_family="SharedMemoryTiling",
                recent_positive_mutations=[{"mutation_family": "SharedMemoryTiling", "single_change_focus": "wider row tile"}],
                recent_negative_mutations=[{"mutation_family": "Reduce_Live_Ranges", "single_change_focus": "trim registers"}],
                plateau_detected=True,
                lineage_plateau_depth=3,
            ),
        )
        refresh_memory_profile(task, feedback, top_k=3)
        assert task.memory_profile is not None
        self.assertTrue(task.memory_profile.feedback_digest["plateau_detected"])
        self.assertIn("SharedMemoryTiling", task.memory_profile.preferred_methods)
        self.assertIn("Reduce_Live_Ranges", task.memory_profile.blocked_methods)
        self.assertIn("wider row tile", task.memory_profile.feedback_digest["recent_positive_focuses"])
        self.assertEqual(task.strategy_portfolio.strategies[0].strategy_id, "strategy_01")

    def test_memory_disables_itself_without_machine_check(self):
        tmp = self._new_tmp_dir()
        try:
            self._write_layernorm_problem(tmp)
            task = self.loader.load_official_problem(tmp, 1, 40, backend="cuda")
            task.diagnostics_profile = TaskDiagnostics(
                enabled=False,
                mode="machine_check_v1",
                notes=["ncu unavailable"],
            )
            task.memory_profile = build_memory_profile(task, enabled=True, max_primary_cards=4, max_challenger_cards=3)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        assert task.memory_profile is not None
        self.assertFalse(task.memory_profile.enabled)
        self.assertIn("ncu unavailable", " ".join(task.memory_profile.notes))


if __name__ == "__main__":
    unittest.main()
