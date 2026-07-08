from __future__ import annotations

from typing import Any

from ..diagnostics import ncu_profile_to_prompt_dict
from .schema import PhaseTransitionSummary


def phase_transition_to_prompt_dict(summary: PhaseTransitionSummary | None, *, attempt_limit: int = 8) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        'source_phase': summary.source_phase,
        'target_phase': summary.target_phase,
        'split_attempts': summary.split_attempts,
        'trigger_attempt': summary.trigger_attempt,
        'root': None
        if summary.root is None
        else {
            'node_id': summary.root.node_id,
            'strategy_name': summary.root.strategy_name,
            'runtime': summary.root.runtime,
            'speedup': summary.root.speedup,
            'plan_summary': summary.root.plan_summary,
            'changed_regions': list(summary.root.changed_regions),
            'lineage': list(summary.root.lineage),
        },
        'selected': None
        if summary.selected is None
        else {
            'node_id': summary.selected.node_id,
            'strategy_name': summary.selected.strategy_name,
            'runtime': summary.selected.runtime,
            'speedup': summary.selected.speedup,
            'plan_summary': summary.selected.plan_summary,
            'changed_regions': list(summary.selected.changed_regions),
            'lineage': list(summary.selected.lineage),
        },
        'attempts': [
            {
                'node_id': attempt.node_id,
                'parent_id': attempt.parent_id,
                'strategy_name': attempt.strategy_name,
                'attempt_mode': attempt.attempt_mode,
                'mutation_family': attempt.mutation_family,
                'single_change_focus': attempt.single_change_focus,
                'changed_regions': list(attempt.changed_regions),
                'compile_ok': attempt.compile_ok,
                'correct': attempt.correct,
                'runtime': attempt.runtime,
                'speedup': attempt.speedup,
                'failure_stage': attempt.failure_stage,
                'failure_type': attempt.failure_type,
                'plan_summary': attempt.plan_summary,
            }
            for attempt in summary.attempts[:attempt_limit]
        ],
        'trace_summary': summary.trace_summary,
        'root_ncu_profile': ncu_profile_to_prompt_dict(getattr(summary.root_diagnostics, 'ncu_profile', None)),
        'selected_ncu_profile': ncu_profile_to_prompt_dict(getattr(summary.selected_diagnostics, 'ncu_profile', None)),
        'diagnostics_delta': dict(summary.diagnostics_delta),
    }
