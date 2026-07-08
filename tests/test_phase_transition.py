import shutil
import unittest
from pathlib import Path

from stark.io import load_run, save_run
from stark.models import RunResult, SearchNode, StarkConfig
from stark.phases import PhaseAttemptTrace, PhaseCandidateSummary, PhaseTransitionSummary


class PhaseTransitionRoundTripTests(unittest.TestCase):
    def test_run_json_round_trips_phase_transition(self):
        config = StarkConfig(
            max_attempts=10,
            phase_two_enabled=True,
            phase_two_split_attempts=5,
            run_profile='quick',
            search_profile='quick',
            evaluator_profile='quick',
            measurement_profile='quick',
        )
        code = '# <<<IMPROVE:cuda_cu>>>\npass\n# <<<END_IMPROVE>>>\n'
        root = SearchNode(
            node_id='root',
            parent_id=None,
            depth=0,
            code=code,
            origin='root',
            compile_ok=True,
            correct=True,
            runtime=10.0,
            score=1.0,
            speedup=1.0,
            node_status='correct',
        )
        child = SearchNode(
            node_id='n1',
            parent_id='root',
            depth=1,
            code=code,
            origin='plan_code',
            compile_ok=True,
            correct=True,
            runtime=5.0,
            score=5.0,
            speedup=2.0,
            node_status='correct',
        )
        root.child_ids = ['n1']
        result = RunResult(
            task_name='phase-transition-task',
            config=config,
            best_node_id='n1',
            leaderboard=['n1'],
            nodes={'root': root, 'n1': child},
            selection_history=['root', 'n1'],
            stats={'best_speedup': 2.0},
            phase_transition=PhaseTransitionSummary(
                source_phase=1,
                target_phase=2,
                split_attempts=5,
                trigger_attempt=5,
                root=PhaseCandidateSummary(node_id='root', runtime=10.0, speedup=1.0),
                selected=PhaseCandidateSummary(
                    node_id='n1',
                    strategy_name='strategy_01',
                    runtime=5.0,
                    speedup=2.0,
                    changed_regions=['cuda_cu'],
                    lineage=['root', 'n1'],
                ),
                attempts=[
                    PhaseAttemptTrace(
                        node_id='n1',
                        parent_id='root',
                        strategy_name='strategy_01',
                        attempt_mode='explore',
                        changed_regions=['cuda_cu'],
                        compile_ok=True,
                        correct=True,
                        runtime=5.0,
                        speedup=2.0,
                    )
                ],
                diagnostics_delta={'kernel_time_ms': -5.0},
            ),
        )
        tmpdir = Path(__file__).resolve().parents[1] / 'runs' / '.tmp_test_phase_transition'
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            run_path = save_run(result, tmpdir)
            reloaded = load_run(run_path)
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

        self.assertTrue(reloaded.config.phase_two_enabled)
        self.assertEqual(reloaded.config.phase_two_split_attempts, 5)
        self.assertIsNotNone(reloaded.phase_transition)
        self.assertEqual(reloaded.phase_transition.selected.node_id, 'n1')
        self.assertEqual(reloaded.phase_transition.selected.changed_regions, ['cuda_cu'])
        self.assertEqual(reloaded.phase_transition.attempts[0].attempt_mode, 'explore')
        self.assertEqual(reloaded.phase_transition.diagnostics_delta['kernel_time_ms'], -5.0)
