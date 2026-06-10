import shutil
import time
import unittest
from pathlib import Path

from stark.core.workflow import run_stark
from stark.diagnostics.schema import MachineCheckProfile, TaskDiagnostics
from stark.deliberation.merge import apply_strategy_reviews, merge_strategy_proposals
from stark.deliberation.runner import MultiModelDeliberationRunner
from stark.deliberation.schema import DeliberationStrategy, ModelProposal, ModelReview
from stark.demo import build_demo_tasks
from stark.evaluation import DemoEvaluator
from stark.io import load_run, save_run
from stark.models import StarkConfig
from stark.providers import MockProvider


class DeliberationTests(unittest.TestCase):
    def test_merge_deduplicates_and_limits_strategies(self):
        proposals = [
            ModelProposal(
                provider_name="mock_a",
                strategies=[
                    DeliberationStrategy(
                        strategy_id="",
                        intent="fusion",
                        summary="Fuse the elementwise chain",
                        target_anchors=["forward_stmt_1"],
                        source_models=["mock_a"],
                        model_scores={"mock_a": 4},
                    )
                ],
            ),
            ModelProposal(
                provider_name="mock_b",
                strategies=[
                    DeliberationStrategy(
                        strategy_id="",
                        intent="fusion",
                        summary="Fuse the elementwise chain",
                        target_anchors=["forward_stmt_1"],
                        source_models=["mock_b"],
                        model_scores={"mock_b": 5},
                    ),
                    DeliberationStrategy(strategy_id="", intent="tiling", summary="Try a tiled variant", source_models=["mock_b"]),
                ],
            ),
        ]
        portfolio = merge_strategy_proposals(proposals, max_strategies=1)
        self.assertEqual(len(portfolio.strategies), 1)
        self.assertEqual(portfolio.strategies[0].strategy_id, "strategy_01")
        self.assertEqual(set(portfolio.strategies[0].source_models), {"mock_a", "mock_b"})

    def test_review_scores_are_attached(self):
        proposal = ModelProposal(
            provider_name="mock_a",
            strategies=[DeliberationStrategy(strategy_id="", intent="fusion", summary="Fuse ops", source_models=["mock_a"])],
        )
        portfolio = merge_strategy_proposals([proposal], max_strategies=10)
        apply_strategy_reviews(portfolio, [ModelReview(provider_name="mock_b", scores={"strategy_01": 4}, notes={"strategy_01": "good"})])
        self.assertEqual(portfolio.strategies[0].model_scores["mock_b"], 4.0)
        self.assertEqual(portfolio.strategies[0].review_notes["mock_b"], "good")

    def test_runner_tolerates_one_failed_model(self):
        class FailingMock(MockProvider):
            name = "failing_mock"

            def generate_text(self, *args, **kwargs):
                raise RuntimeError("provider down")

        task = build_demo_tasks()[0]
        config = StarkConfig(deliberation_enabled=True, deliberation_providers=["mock_ok", "mock_fail"])
        runner = MultiModelDeliberationRunner(
            providers={"mock_ok": MockProvider(), "mock_fail": FailingMock()},
            max_strategies=10,
            strategies_per_model=2,
        )
        portfolio = runner.run(task, config)
        self.assertTrue(portfolio.enabled)
        self.assertIn("mock_fail", portfolio.proposal_errors)
        self.assertGreaterEqual(len(portfolio.strategies), 1)

    def test_run_json_round_trips_strategy_portfolio(self):
        task = build_demo_tasks()[0]
        config = StarkConfig(max_attempts=1, deliberation_enabled=True, deliberation_profile="quick")
        runner = MultiModelDeliberationRunner(providers={"mock": MockProvider()}, max_strategies=2, strategies_per_model=1)
        task.diagnostics_profile = TaskDiagnostics(
            enabled=True,
            mode="machine_check_v1",
            machine_check_profile=MachineCheckProfile(
                enabled=True,
                status="ok",
                case_id="DEMO_CASE",
                allowed_methods=["Launch_Tuning"],
            ),
        )
        task.strategy_portfolio = runner.run(task, config)
        result = run_stark(task, config, MockProvider(), DemoEvaluator())
        tmpdir = Path(__file__).resolve().parents[1] / "runs" / ".tmp_test_deliberation"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            run_path = save_run(result, tmpdir)
            reloaded = load_run(run_path)
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)
        self.assertIsNotNone(reloaded.strategy_portfolio)
        self.assertGreaterEqual(len(reloaded.strategy_portfolio.strategies), 1)
        self.assertIsNotNone(reloaded.diagnostics_profile)
        self.assertEqual(reloaded.diagnostics_profile.machine_check_profile.case_id, "DEMO_CASE")

    def test_mock_plan_uses_untried_portfolio_strategy(self):
        task = build_demo_tasks()[0]
        task.strategy_catalog = []
        config = StarkConfig(max_attempts=2, deliberation_enabled=True)
        runner = MultiModelDeliberationRunner(providers={"mock_a": MockProvider(), "mock_b": MockProvider()}, max_strategies=2, strategies_per_model=1)
        task.strategy_portfolio = runner.run(task, config)
        result = run_stark(task, config, MockProvider(), DemoEvaluator())
        used = [node.plan_strategy_name for node in result.nodes.values() if node.plan_strategy_name]
        self.assertTrue(any(name and name.startswith("strategy_") for name in used))
    def test_runner_collects_in_parallel(self):
        class SlowMock(MockProvider):
            def __init__(self, delay: float) -> None:
                super().__init__()
                self.delay = delay

            def generate_text(self, *args, **kwargs):
                time.sleep(self.delay)
                return super().generate_text(*args, **kwargs)

        task = build_demo_tasks()[0]
        config = StarkConfig(deliberation_enabled=True, deliberation_providers=["mock_a", "mock_b", "mock_c"])
        runner = MultiModelDeliberationRunner(
            providers={
                "mock_a": SlowMock(0.2),
                "mock_b": SlowMock(0.2),
                "mock_c": SlowMock(0.2),
            },
            max_strategies=3,
            strategies_per_model=1,
        )
        started = time.time()
        portfolio = runner.run(task, config)
        elapsed = time.time() - started
        self.assertTrue(portfolio.enabled)
        self.assertLess(elapsed, 1.0)
        starts = [event for event in runner.last_events if event.status == "start"]
        oks = [event for event in runner.last_events if event.status == "ok"]
        self.assertEqual(len(starts), 6)
        self.assertEqual(len(oks), 6)
