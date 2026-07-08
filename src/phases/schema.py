from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..diagnostics.schema import TaskDiagnostics, task_diagnostics_from_dict, task_diagnostics_to_dict


@dataclass
class PhaseCandidateSummary:
    node_id: str
    strategy_name: str | None = None
    runtime: float | None = None
    speedup: float | None = None
    plan_summary: str | None = None
    changed_regions: list[str] = field(default_factory=list)
    lineage: list[str] = field(default_factory=list)


@dataclass
class PhaseAttemptTrace:
    node_id: str
    parent_id: str | None = None
    strategy_name: str | None = None
    attempt_mode: str | None = None
    mutation_family: str | None = None
    single_change_focus: str | None = None
    changed_regions: list[str] = field(default_factory=list)
    compile_ok: bool = False
    correct: bool = False
    runtime: float | None = None
    speedup: float | None = None
    failure_stage: str | None = None
    failure_type: str | None = None
    plan_summary: str | None = None


@dataclass
class PhaseTransitionSummary:
    source_phase: int = 1
    target_phase: int = 2
    split_attempts: int = 5
    trigger_attempt: int = 5
    root: PhaseCandidateSummary | None = None
    selected: PhaseCandidateSummary | None = None
    attempts: list[PhaseAttemptTrace] = field(default_factory=list)
    trace_summary: str | None = None
    root_diagnostics: TaskDiagnostics | None = None
    selected_diagnostics: TaskDiagnostics | None = None
    diagnostics_delta: dict[str, Any] = field(default_factory=dict)


def _candidate_to_dict(candidate: PhaseCandidateSummary | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        'node_id': candidate.node_id,
        'strategy_name': candidate.strategy_name,
        'runtime': candidate.runtime,
        'speedup': candidate.speedup,
        'plan_summary': candidate.plan_summary,
        'changed_regions': list(candidate.changed_regions),
        'lineage': list(candidate.lineage),
    }


def _candidate_from_dict(payload: dict[str, Any] | None) -> PhaseCandidateSummary | None:
    if not payload:
        return None
    return PhaseCandidateSummary(
        node_id=str(payload.get('node_id', '')),
        strategy_name=payload.get('strategy_name'),
        runtime=payload.get('runtime'),
        speedup=payload.get('speedup'),
        plan_summary=payload.get('plan_summary'),
        changed_regions=[str(item) for item in payload.get('changed_regions', [])],
        lineage=[str(item) for item in payload.get('lineage', [])],
    )


def phase_transition_to_dict(summary: PhaseTransitionSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        'source_phase': summary.source_phase,
        'target_phase': summary.target_phase,
        'split_attempts': summary.split_attempts,
        'trigger_attempt': summary.trigger_attempt,
        'root': _candidate_to_dict(summary.root),
        'selected': _candidate_to_dict(summary.selected),
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
            for attempt in summary.attempts
        ],
        'trace_summary': summary.trace_summary,
        'root_diagnostics': task_diagnostics_to_dict(summary.root_diagnostics),
        'selected_diagnostics': task_diagnostics_to_dict(summary.selected_diagnostics),
        'diagnostics_delta': dict(summary.diagnostics_delta),
    }


def phase_transition_from_dict(payload: dict[str, Any] | None) -> PhaseTransitionSummary | None:
    if not payload:
        return None
    return PhaseTransitionSummary(
        source_phase=int(payload.get('source_phase', 1)),
        target_phase=int(payload.get('target_phase', 2)),
        split_attempts=int(payload.get('split_attempts', 5)),
        trigger_attempt=int(payload.get('trigger_attempt', payload.get('split_attempts', 5))),
        root=_candidate_from_dict(payload.get('root')),
        selected=_candidate_from_dict(payload.get('selected')),
        attempts=[
            PhaseAttemptTrace(
                node_id=str(item.get('node_id', '')),
                parent_id=item.get('parent_id'),
                strategy_name=item.get('strategy_name'),
                attempt_mode=item.get('attempt_mode'),
                mutation_family=item.get('mutation_family'),
                single_change_focus=item.get('single_change_focus'),
                changed_regions=[str(region) for region in item.get('changed_regions', [])],
                compile_ok=bool(item.get('compile_ok', False)),
                correct=bool(item.get('correct', False)),
                runtime=item.get('runtime'),
                speedup=item.get('speedup'),
                failure_stage=item.get('failure_stage'),
                failure_type=item.get('failure_type'),
                plan_summary=item.get('plan_summary'),
            )
            for item in payload.get('attempts', [])
        ],
        trace_summary=payload.get('trace_summary'),
        root_diagnostics=task_diagnostics_from_dict(payload.get('root_diagnostics')),
        selected_diagnostics=task_diagnostics_from_dict(payload.get('selected_diagnostics')),
        diagnostics_delta=dict(payload.get('diagnostics_delta') or {}),
    )
