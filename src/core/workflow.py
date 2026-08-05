"""Main STARK workflow runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from ..agents import CodeAgent, DebugAgent, PlanAgent
from ..diagnostics import build_task_diagnostics
from ..feedback import collect_feedback_state
from ..models import AgentContext, AnchorEdit, EvaluationResult, PlanProposal, RunResult, SearchNode, StarkConfig, TaskSpec
from ..phases import PhaseAttemptTrace, PhaseCandidateSummary, PhaseTransitionSummary
from ..providers.errors import is_transient_provider_error
from ..utils import extract_anchor_names, preserve_anchor_scaffold, shorten_runtime
from .candidate import normalize_candidate
from .context import build_code_context, build_debug_context, build_plan_context, snapshot_node
from .regions import preserve_region_scaffold
from .static_check import check_candidate_static
from .tree import TreeMemory


def _node_status_from_evaluation(evaluation: EvaluationResult) -> str:
    if evaluation.failure_stage == "compile":
        return "compile_fail"
    if evaluation.failure_stage == "runtime":
        return "runtime_fail"
    if evaluation.failure_stage == "correctness":
        return "correctness_fail"
    return "correct"


def _evaluation_to_root_node(code: str, evaluation: EvaluationResult) -> SearchNode:
    return SearchNode(
        node_id="root",
        parent_id=None,
        depth=0,
        code=code,
        origin="root",
        compile_ok=evaluation.compile_ok,
        correct=evaluation.correct,
        runtime=evaluation.runtime,
        score=evaluation.score,
        logs=list(evaluation.logs),
        failure_type=evaluation.failure_type,
        node_status=_node_status_from_evaluation(evaluation),
        latest_failure_stage=None if evaluation.failure_stage == "none" else evaluation.failure_stage,
        reference_runtime=evaluation.reference_runtime,
        speedup=evaluation.speedup,
        reference_runtimes=dict(evaluation.reference_runtimes),
        speedups=dict(evaluation.speedups),
        primary_reference=evaluation.primary_reference,
    )


def _has_valid_anchor_edits(code: str, anchor_edits: list[AnchorEdit]) -> bool:
    available_anchors = set(extract_anchor_names(code))
    if not anchor_edits or not available_anchors:
        return False
    for edit in anchor_edits:
        if edit.anchor_name not in available_anchors:
            return False
        if edit.operation not in {"replace", "append"}:
            return False
    return True


def _invalid_anchor_evaluation(code: str, proposal: PlanProposal) -> EvaluationResult:
    available = ", ".join(extract_anchor_names(code)) or "none"
    requested = ", ".join(f"{edit.anchor_name}:{edit.operation}" for edit in proposal.anchor_edits) or "none"
    return EvaluationResult(
        compile_ok=False,
        correct=False,
        runtime=None,
        score=float("inf"),
        logs=[f"invalid_anchor_edit: available={available}; requested={requested}"],
        failure_type="invalid_anchor_edit",
        failure_stage="compile",
    )


def _guard_failure(failure_type: str, logs: list[str]) -> EvaluationResult:
    return EvaluationResult(
        compile_ok=False,
        correct=False,
        runtime=None,
        score=float("inf"),
        logs=list(logs),
        failure_type=failure_type,
        failure_stage="compile",
    )


def _merge_logs(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for entry in group or []:
            if entry and entry not in merged:
                merged.append(entry)
    return merged


def _attach_logs(evaluation: EvaluationResult, prefix_logs: list[str]) -> EvaluationResult:
    if not prefix_logs:
        return evaluation
    evaluation.logs = _merge_logs(prefix_logs, evaluation.logs)
    return evaluation


def _prepare_candidate_for_evaluation(
    task: TaskSpec,
    parent_code: str,
    raw_candidate: str,
    *,
    allowed_regions: set[str] | None = None,
    frozen_regions: set[str] | None = None,
) -> tuple[str, list[str], EvaluationResult | None]:
    normalized = normalize_candidate(
        parent_code,
        raw_candidate,
        allowed_regions=allowed_regions,
        frozen_regions=frozen_regions,
    )
    if not normalized.ok:
        return normalized.code, [], _guard_failure(normalized.failure_type or "invalid_candidate", normalized.logs)
    static_result = check_candidate_static(normalized.code, backend=task.backend)
    if not static_result.ok:
        return normalized.code, normalized.logs, _guard_failure(
            static_result.failure_type or "static_check_failed",
            _merge_logs(normalized.logs, static_result.logs),
        )
    return normalized.code, normalized.logs, None


def _broken_anchor_evaluation(parent_code: str, candidate_code: str) -> EvaluationResult:
    expected = extract_anchor_names(parent_code)
    observed = extract_anchor_names(candidate_code)
    return EvaluationResult(
        compile_ok=False,
        correct=False,
        runtime=None,
        score=float("inf"),
        logs=[f"broken_anchor_markers: expected={expected}; observed={observed}"],
        failure_type="broken_anchor_markers",
        failure_stage="compile",
    )


def _anchors_preserved(parent_code: str, candidate_code: str) -> bool:
    return preserve_region_scaffold(parent_code, candidate_code)


def _evaluate_marker_drift(task: TaskSpec, config: StarkConfig, evaluator, parent_code: str, candidate_code: str) -> EvaluationResult:
    del task, config, evaluator
    expected = extract_anchor_names(parent_code)
    observed = extract_anchor_names(candidate_code)
    return EvaluationResult(
        compile_ok=False,
        correct=False,
        runtime=None,
        score=float("inf"),
        logs=[f"anchor_marker_drift: expected={expected}; observed={observed}"],
        failure_type="anchor_marker_drift",
        failure_stage="compile",
    )


def _build_debug_proposal(node: SearchNode) -> PlanProposal:
    return PlanProposal(
        strategy_name=node.plan_strategy_name or "debug",
        strategy_summary=node.plan_summary or "Repair previous candidate",
        anchor_edits=list(node.anchor_edits),
        expected_gain="Recover a failing candidate with a local fix.",
        risk_notes="Debug route applies the smallest viable local repair.",
        mode="refine",
        attempt_mode="mutate_champion",
        target_node_id=node.node_id,
        target_anchors=[edit.anchor_name for edit in node.anchor_edits],
        frozen_anchors=[],
        change_budget="small",
        must_preserve=["Keep the current working structure intact while repairing the failure."],
        reason_against_rewrite="Debug should make the smallest local fix.",
        performance_hypothesis="Fix the local failure without changing the successful structure.",
        single_change_focus="small_local_repair",
        mutation_family="debug_local_repair",
    )


def _record_failure(stats: dict, evaluation: EvaluationResult) -> None:
    if evaluation.failure_type:
        failure_counts = stats.setdefault("failure_counts", {})
        failure_counts[evaluation.failure_type] = failure_counts.get(evaluation.failure_type, 0) + 1
    if evaluation.failure_stage and evaluation.failure_stage != "none":
        stage_counts = stats.setdefault("failure_stage_counts", {})
        stage_counts[evaluation.failure_stage] = stage_counts.get(evaluation.failure_stage, 0) + 1


def _new_stats() -> dict:
    return {
        "attempt_count": 0,
        "plan_attempts": 0,
        "debug_attempts": 0,
        "failure_counts": {},
        "failure_stage_counts": {},
        "invalid_proposals": 0,
        "attempt_mode_counts": {},
        "provider_transient_error_count": 0,
        "provider_transient_budget_exhausted_count": 0,
        "provider_transient_errors_by_stage": {},
        "provider_transient_errors_by_origin": {},
        "provider_transient_errors": [],
    }


def _new_debug_stats() -> dict:
    return {
        "total_attempts": 0,
        "per_node": {},
    }


WorkflowInitialState = tuple[TreeMemory, EvaluationResult, dict, dict]


@dataclass
class StageRunState:
    tree: TreeMemory
    root_eval: EvaluationResult
    stats: dict[str, Any]
    debug_stats: dict[str, Any]
    selection_history: list[str]
    selection_reasons: list[str]
    selection_exclusions: list[dict[str, str]]
    created_node_ids: list[str]
    feedback_state: Any
    deliberation_round: int
    last_upgrade_champion_id: str | None = None


def _initialize_tree(task: TaskSpec, config: StarkConfig, evaluator) -> WorkflowInitialState:
    root_eval = evaluator.evaluate(task, task.source_code, config)
    tree = TreeMemory(_evaluation_to_root_node(task.source_code, root_eval), config)
    tree.update_leaderboard(tree.root_id, config)
    stats = _new_stats()
    debug_stats = _new_debug_stats()
    _record_failure(stats, root_eval)
    return tree, root_eval, stats, debug_stats


def _root_only_context(tree: TreeMemory, role: str, feedback_state=None) -> AgentContext:
    root = snapshot_node(tree, tree.root_id)
    return AgentContext(role=role, current=root, root=root, related=[], leaders=[], failure=None, feedback_state=feedback_state)


def bootstrap_stark_root(task: TaskSpec, config: StarkConfig, evaluator) -> WorkflowInitialState:
    if config.verbose:
        print(f"[workflow] root_evaluation_start task={task.name}", flush=True)
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
    if config.verbose:
        print(
            f"[workflow] root_evaluation_done status={tree.get_node(tree.root_id).status} "
            f"runtime={shorten_runtime(root_eval.runtime)}",
            flush=True,
        )
    return tree, root_eval, stats, debug_stats


def _record_attempt_mode(stats: dict, mode: str) -> None:
    counts = stats.setdefault("attempt_mode_counts", {})
    counts[mode] = counts.get(mode, 0) + 1


def _provider_transient_error_budget(attempt_budget: int) -> int:
    return max(6, int(attempt_budget) * 4)


def _rollback_transient_selection(tree: TreeMemory, selected_id: str) -> None:
    if selected_id in tree.nodes:
        node = tree.get_node(selected_id)
        node.selected_count = max(0, int(getattr(node, "selected_count", 0)) - 1)
    if getattr(tree, "selection_exclusion_history", None):
        tree.selection_exclusion_history.pop()


def _record_provider_transient_error(
    stats: dict[str, Any],
    *,
    stage_name: str,
    attempt_index: int,
    selected_id: str,
    origin: str,
    exc: BaseException,
    error_index: int,
    error_budget: int,
) -> None:
    stats["provider_transient_error_count"] = int(stats.get("provider_transient_error_count", 0)) + 1
    by_stage = stats.setdefault("provider_transient_errors_by_stage", {})
    by_stage[stage_name] = int(by_stage.get(stage_name, 0)) + 1
    by_origin = stats.setdefault("provider_transient_errors_by_origin", {})
    by_origin[origin] = int(by_origin.get(origin, 0)) + 1
    errors = stats.setdefault("provider_transient_errors", [])
    if len(errors) < 50:
        errors.append(
            {
                "stage": stage_name,
                "attempt_index": attempt_index,
                "selected_node_id": selected_id,
                "origin": origin,
                "error_index": error_index,
                "error_budget": error_budget,
                "message": _short_error_message(exc),
            }
        )


def _short_error_message(exc: BaseException, limit: int = 500) -> str:
    message = " ".join(str(exc).split())
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _attempt_mode_for_selected_node(
    tree: TreeMemory,
    selected_id: str,
    selection_reason: str,
    feedback_state,
) -> str:
    if selected_id == tree.root_id:
        return "explore"
    node = tree.get_node(selected_id)
    if node.is_failure:
        return "debug"

    champion_id = getattr(feedback_state, "current_champion_id", None) if feedback_state is not None else None
    champion_speedup = getattr(feedback_state, "current_champion_speedup", None) if feedback_state is not None else None
    if champion_id and champion_id in tree.nodes and champion_speedup is None:
        champion_speedup = tree.get_node(champion_id).speedup

    actionable_champion = (
        champion_id is not None
        and champion_id in tree.nodes
        and champion_id != tree.root_id
        and champion_speedup is not None
        and champion_speedup > 1.0
    )
    if not actionable_champion:
        return "explore"

    champion_subtree = tree.subtree_ids(champion_id)
    champion_lineage = set(getattr(getattr(feedback_state, "champion", None), "lineage", []) or [])
    if selected_id == champion_id:
        return "best_lineage_push" if node.child_ids else "mutate_champion"
    if selected_id in champion_subtree or selected_id in champion_lineage:
        return "mutate_champion"
    return "challenger"


def _champion_upgrade_context(tree: TreeMemory, feedback_state) -> tuple[dict[str, object] | None, str | None]:
    champion_id = getattr(feedback_state, "current_champion_id", None) if feedback_state is not None else None
    if not champion_id or champion_id not in tree.nodes:
        return None, None
    champion = tree.get_node(champion_id)
    champion_snapshot = snapshot_node(tree, champion_id)
    summary = {
        "node_id": champion_snapshot.node_id,
        "speedup": champion_snapshot.speedup,
        "strategy_name": champion.plan_strategy_name,
        "plan_mode": champion.plan_mode,
        "mutation_family": champion.mutation_family,
        "single_change_focus": champion.single_change_focus,
        "anchor_names": [edit.anchor_name for edit in champion.anchor_edits],
        "failure_log_excerpt": champion_snapshot.failure_log_excerpt,
    }
    return summary, champion.code


def _release_provider(provider) -> None:
    close_fn = getattr(provider, 'close', None)
    if callable(close_fn):
        close_fn()


def _search_only_context(tree: TreeMemory, node_id: str) -> AgentContext:
    current = snapshot_node(tree, node_id)
    root = snapshot_node(tree, tree.root_id)
    return AgentContext(
        role="search",
        current=current,
        root=root,
        related=[],
        leaders=[],
        failure=current if current.latest_failure_stage is not None else None,
    )


def _evaluate_plan_code(
    task: TaskSpec,
    config: StarkConfig,
    evaluator,
    code_agent,
    selected_node: SearchNode,
    proposal: PlanProposal,
    code_context: AgentContext,
    stats: dict,
) -> tuple[str, EvaluationResult]:
    if not _has_valid_anchor_edits(selected_node.code, proposal.anchor_edits):
        stats["invalid_proposals"] += 1
        return selected_node.code, _invalid_anchor_evaluation(selected_node.code, proposal)
    raw_candidate = code_agent.run(task, selected_node, proposal, code_context)
    candidate_code, candidate_logs, guard_evaluation = _prepare_candidate_for_evaluation(
        task,
        selected_node.code,
        raw_candidate,
        allowed_regions=set(proposal.target_anchors or [edit.anchor_name for edit in proposal.anchor_edits]),
        frozen_regions=set(proposal.frozen_anchors),
    )
    if guard_evaluation is not None:
        return candidate_code, guard_evaluation
    if not _anchors_preserved(selected_node.code, candidate_code):
        return candidate_code, _attach_logs(
            _evaluate_marker_drift(task, config, evaluator, selected_node.code, candidate_code),
            candidate_logs,
        )
    return candidate_code, _attach_logs(evaluator.evaluate(task, candidate_code, config), candidate_logs)


def _evaluate_debug(
    task: TaskSpec,
    config: StarkConfig,
    evaluator,
    debug_agent,
    selected_node: SearchNode,
    debug_context: AgentContext,
) -> tuple[PlanProposal, str, EvaluationResult]:
    raw_candidate = debug_agent.run(task, selected_node, debug_context)
    proposal = _build_debug_proposal(selected_node)
    candidate_code, candidate_logs, guard_evaluation = _prepare_candidate_for_evaluation(
        task,
        selected_node.code,
        raw_candidate,
        allowed_regions=set(edit.anchor_name for edit in proposal.anchor_edits),
        frozen_regions=set(),
    )
    if guard_evaluation is not None:
        return proposal, candidate_code, guard_evaluation
    if not _anchors_preserved(selected_node.code, candidate_code):
        return proposal, candidate_code, _attach_logs(
            _evaluate_marker_drift(task, config, evaluator, selected_node.code, candidate_code),
            candidate_logs,
        )
    return proposal, candidate_code, _attach_logs(evaluator.evaluate(task, candidate_code, config), candidate_logs)


def _node_to_evaluation(node: SearchNode) -> EvaluationResult:
    return EvaluationResult(
        compile_ok=node.compile_ok,
        correct=node.correct,
        runtime=node.runtime,
        score=node.score,
        logs=list(node.logs),
        failure_type=node.failure_type,
        failure_stage=node.latest_failure_stage or "none",
        reference_runtime=node.reference_runtime,
        speedup=node.speedup,
        reference_runtimes=dict(node.reference_runtimes),
        speedups=dict(node.speedups),
        primary_reference=node.primary_reference,
    )


def _initial_state_from_root_candidate(node: SearchNode, config: StarkConfig) -> WorkflowInitialState:
    root_node = deepcopy(node)
    root_node.node_id = "root"
    root_node.parent_id = None
    root_node.depth = 0
    root_node.child_ids = []
    root_node.selected_count = 0
    root_node.selection_reason = None
    root_node.prune_reason = None
    root_node.debug_attempts = 0
    root_node.origin = "phase2_root"
    root_eval = _node_to_evaluation(node)
    tree = TreeMemory(root_node, config)
    tree.update_leaderboard(tree.root_id, config)
    stats = _new_stats()
    debug_stats = _new_debug_stats()
    _record_failure(stats, root_eval)
    return tree, root_eval, stats, debug_stats


def _node_lineage(tree: TreeMemory, node_id: str) -> list[str]:
    lineage: list[str] = []
    current_id: str | None = node_id
    while current_id is not None and current_id in tree.nodes:
        lineage.append(current_id)
        current_id = tree.get_node(current_id).parent_id
    lineage.reverse()
    return lineage


def _changed_regions(node: SearchNode) -> list[str]:
    regions: list[str] = []
    for edit in node.anchor_edits:
        if edit.anchor_name not in regions:
            regions.append(edit.anchor_name)
    return regions


def _phase_candidate_summary(tree: TreeMemory, node_id: str) -> PhaseCandidateSummary:
    node = tree.get_node(node_id)
    return PhaseCandidateSummary(
        node_id=node.node_id,
        strategy_name=node.plan_strategy_name,
        runtime=node.runtime,
        speedup=node.speedup,
        plan_summary=node.plan_summary,
        changed_regions=_changed_regions(node),
        lineage=_node_lineage(tree, node_id),
    )


def _phase_attempt_trace(tree: TreeMemory, node_id: str) -> PhaseAttemptTrace:
    node = tree.get_node(node_id)
    return PhaseAttemptTrace(
        node_id=node.node_id,
        parent_id=node.parent_id,
        strategy_name=node.plan_strategy_name,
        attempt_mode=node.plan_mode,
        mutation_family=node.mutation_family,
        single_change_focus=node.single_change_focus,
        changed_regions=_changed_regions(node),
        compile_ok=node.compile_ok,
        correct=node.correct,
        runtime=node.runtime,
        speedup=node.speedup,
        failure_stage=node.latest_failure_stage,
        failure_type=node.failure_type,
        plan_summary=node.plan_summary,
    )


def _trace_summary(tree: TreeMemory, node_ids: list[str]) -> str | None:
    improved: list[str] = []
    flat: list[str] = []
    failed: list[str] = []
    for node_id in node_ids:
        if node_id not in tree.nodes:
            continue
        node = tree.get_node(node_id)
        label = node.single_change_focus or node.plan_strategy_name or node.node_id
        if not node.compile_ok or not node.correct:
            failed.append(f"{node.node_id}:{label}:{node.failure_type or node.latest_failure_stage or 'failure'}")
            continue
        parent = tree.get_node(node.parent_id) if node.parent_id and node.parent_id in tree.nodes else None
        parent_speedup = parent.speedup if parent is not None else None
        if (
            isinstance(node.speedup, (int, float))
            and isinstance(parent_speedup, (int, float))
            and node.speedup > parent_speedup + 1e-6
        ):
            improved.append(f"{node.node_id}:{label}:{node.speedup:.3f}x")
        else:
            speed_text = f"{node.speedup:.3f}x" if isinstance(node.speedup, (int, float)) else "n/a"
            flat.append(f"{node.node_id}:{label}:{speed_text}")
    sections: list[str] = []
    if improved:
        sections.append("effective=" + "; ".join(improved[:4]))
    if flat:
        sections.append("flat=" + "; ".join(flat[:4]))
    if failed:
        sections.append("failed=" + "; ".join(failed[:4]))
    return " | ".join(sections) if sections else None


def _phase_diagnostics_delta(root_diagnostics, selected_diagnostics) -> dict[str, Any]:
    root_metrics = dict(getattr(getattr(root_diagnostics, "ncu_profile", None), "raw_metrics", {}) or {})
    selected_metrics = dict(getattr(getattr(selected_diagnostics, "ncu_profile", None), "raw_metrics", {}) or {})
    delta: dict[str, Any] = {}
    for key in sorted(set(root_metrics) | set(selected_metrics)):
        root_value = root_metrics.get(key)
        selected_value = selected_metrics.get(key)
        if isinstance(root_value, (int, float)) and isinstance(selected_value, (int, float)):
            delta[key] = selected_value - root_value
        elif selected_value is not None and root_value != selected_value:
            delta[key] = {"root": root_value, "selected": selected_value}
    return delta


def _select_phase_two_seed(tree: TreeMemory, node_ids: list[str]) -> str | None:
    candidates: list[SearchNode] = []
    for node_id in node_ids:
        node = tree.nodes.get(node_id)
        if node is None or not node.compile_ok or not node.correct:
            continue
        if node.node_id == tree.root_id:
            continue
        if not isinstance(node.runtime, (int, float)):
            continue
        candidates.append(node)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.runtime, item.depth, item.node_id))
    return candidates[0].node_id


def _build_phase_transition_summary(
    task: TaskSpec,
    config: StarkConfig,
    stage: StageRunState,
    selected_node_id: str,
) -> PhaseTransitionSummary:
    trigger_attempt = max(1, min(int(getattr(config, "phase_two_split_attempts", 5)), len(stage.created_node_ids)))
    return PhaseTransitionSummary(
        source_phase=1,
        target_phase=2,
        split_attempts=trigger_attempt,
        trigger_attempt=trigger_attempt,
        root=_phase_candidate_summary(stage.tree, stage.tree.root_id),
        selected=_phase_candidate_summary(stage.tree, selected_node_id),
        attempts=[_phase_attempt_trace(stage.tree, node_id) for node_id in stage.created_node_ids],
        trace_summary=_trace_summary(stage.tree, stage.created_node_ids),
        root_diagnostics=task.diagnostics_profile,
        selected_diagnostics=None,
        diagnostics_delta={},
    )


def _feedback_state_for_nodes(
    config: StarkConfig,
    nodes: dict[str, SearchNode],
    leaderboard: list[str],
    pruned_nodes: dict[str, str],
):
    root = deepcopy(nodes["root"])
    temp_tree = TreeMemory(root, config)
    temp_tree.nodes = deepcopy(nodes)
    temp_tree.leaderboard = list(leaderboard)
    temp_tree.pruned_nodes = dict(pruned_nodes)
    return collect_feedback_state(
        temp_tree,
        plateau_delta_threshold=config.plateau_delta_threshold,
        plateau_window=config.plateau_window,
    )


def _best_node_id_from_nodes(nodes: dict[str, SearchNode], fallback_root_id: str = "root") -> str:
    correct_nodes = [node for node in nodes.values() if node.compile_ok and node.correct]
    if not correct_nodes:
        return fallback_root_id
    correct_nodes.sort(key=lambda node: (node.score, node.depth, node.node_id))
    return correct_nodes[0].node_id


def _combine_count_maps(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] = merged[key] + value
        elif isinstance(value, (int, float)) and key not in merged:
            merged[key] = value
        else:
            merged[key] = deepcopy(value)
    return merged


def _combine_stats(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key in (
        "attempt_count",
        "plan_attempts",
        "debug_attempts",
        "invalid_proposals",
        "provider_transient_error_count",
        "provider_transient_budget_exhausted_count",
    ):
        merged[key] = int(left.get(key, 0)) + int(right.get(key, 0))
    for key in (
        "failure_counts",
        "failure_stage_counts",
        "attempt_mode_counts",
        "provider_transient_errors_by_stage",
        "provider_transient_errors_by_origin",
    ):
        merged[key] = _combine_count_maps(dict(left.get(key, {})), dict(right.get(key, {})))
    merged["provider_transient_errors"] = (
        list(left.get("provider_transient_errors", [])) + list(right.get("provider_transient_errors", []))
    )[:50]
    return merged


def _combine_debug_stats(
    left: dict[str, Any],
    right: dict[str, Any],
    phase_two_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    merged = {
        "total_attempts": int(left.get("total_attempts", 0)) + int(right.get("total_attempts", 0)),
        "per_node": dict(left.get("per_node", {})),
    }
    mapping = phase_two_mapping or {}
    for node_id, count in dict(right.get("per_node", {})).items():
        mapped_id = mapping.get(node_id, node_id)
        merged["per_node"][mapped_id] = merged["per_node"].get(mapped_id, 0) + count
    return merged


def _clone_phase_two_nodes(
    tree: TreeMemory,
    attach_parent_id: str,
    attach_depth: int,
) -> tuple[dict[str, SearchNode], dict[str, str]]:
    original_nodes = tree.clone_nodes()
    mapping = {node_id: f"phase2_{node_id}" for node_id in original_nodes}
    cloned: dict[str, SearchNode] = {}
    for node_id, node in original_nodes.items():
        node.node_id = mapping[node_id]
        if node_id == tree.root_id:
            node.parent_id = attach_parent_id
            node.depth = attach_depth
            node.origin = "phase2_root"
        else:
            node.parent_id = mapping.get(node.parent_id) if node.parent_id is not None else attach_parent_id
            node.depth = attach_depth + node.depth
        node.child_ids = [mapping[child_id] for child_id in node.child_ids]
        cloned[node.node_id] = node
    return cloned, mapping


def _build_run_result(
    task: TaskSpec,
    config: StarkConfig,
    nodes: dict[str, SearchNode],
    leaderboard: list[str],
    leaderboard_history: list[list[str]],
    selection_history: list[str],
    selection_reasons: list[str],
    selection_exclusions: list[dict[str, str]],
    pruned_nodes: dict[str, str],
    stats: dict[str, Any],
    debug_stats: dict[str, Any],
    workflow: str,
) -> RunResult:
    best_node_id = _best_node_id_from_nodes(nodes)
    best_node = nodes[best_node_id]
    stats = dict(stats)
    stats["pruned_count"] = len(pruned_nodes)
    feedback_state = _feedback_state_for_nodes(config, nodes, leaderboard, pruned_nodes)
    return RunResult(
        task_name=task.name,
        config=config,
        best_node_id=best_node_id,
        leaderboard=list(leaderboard),
        nodes=nodes,
        selection_history=selection_history,
        stats=stats,
        leaderboard_history=[list(snapshot) for snapshot in leaderboard_history],
        selection_reasons=selection_reasons,
        selection_exclusions=selection_exclusions,
        pruned_nodes=dict(pruned_nodes),
        debug_stats=debug_stats,
        benchmark_family=task.benchmark_family,
        level=task.level,
        problem_id=task.problem_id,
        backend=task.backend,
        source_origin=task.source_origin,
        source_root=task.source_root,
        workflow=workflow,
        run_profile=config.run_profile,
        search_profile=config.search_profile,
        evaluator_profile=config.evaluator_profile,
        measurement_profile=config.measurement_profile,
        preset=config.preset,
        evaluation_profile=config.evaluation_profile,
        kernelbench_evaluator=config.kernelbench_evaluator,
        grounded_regions=list(task.grounded_regions),
        semantic_profile=task.semantic_profile,
        diagnostics_profile=task.diagnostics_profile,
        strategy_portfolio=task.strategy_portfolio,
        phase_transition=task.phase_transition,
        feedback_state=feedback_state,
        reference_runtimes=dict(best_node.reference_runtimes),
        speedups=dict(best_node.speedups),
        primary_reference=best_node.primary_reference,
    )


def _finalize_single_stage_run(
    task: TaskSpec,
    config: StarkConfig,
    stage: StageRunState,
    workflow: str,
) -> RunResult:
    return _build_run_result(
        task,
        config,
        stage.tree.clone_nodes(),
        list(stage.tree.leaderboard),
        [list(snapshot) for snapshot in stage.tree.leaderboard_history],
        list(stage.selection_history),
        list(stage.selection_reasons),
        [dict(item) for item in stage.selection_exclusions],
        dict(stage.tree.pruned_nodes),
        dict(stage.stats),
        deepcopy(stage.debug_stats),
        workflow=workflow,
    )


def _finalize_two_phase_run(
    task: TaskSpec,
    config: StarkConfig,
    phase1: StageRunState,
    phase2: StageRunState,
    phase1_best_node_id: str,
    workflow: str,
) -> RunResult:
    combined_nodes = phase1.tree.clone_nodes()
    phase1_best = combined_nodes[phase1_best_node_id]
    phase2_nodes, mapping = _clone_phase_two_nodes(
        phase2.tree,
        attach_parent_id=phase1_best_node_id,
        attach_depth=phase1_best.depth + 1,
    )
    phase2_root_id = mapping[phase2.tree.root_id]
    if phase2_root_id not in phase1_best.child_ids:
        phase1_best.child_ids.append(phase2_root_id)
    combined_nodes.update(phase2_nodes)
    combined_leaderboard = [
        node_id
        for node_id in sorted(
            [node.node_id for node in combined_nodes.values() if node.compile_ok and node.correct],
            key=lambda current_id: (
                combined_nodes[current_id].score,
                combined_nodes[current_id].depth,
                current_id,
            ),
        )[: config.leaderboard_size]
    ]
    phase2_history = [[mapping.get(node_id, node_id) for node_id in snapshot] for snapshot in phase2.tree.leaderboard_history]
    selection_history = list(phase1.selection_history) + [mapping.get(node_id, node_id) for node_id in phase2.selection_history]
    selection_reasons = list(phase1.selection_reasons) + list(phase2.selection_reasons)
    selection_exclusions = list(phase1.selection_exclusions) + [
        {mapping.get(node_id, node_id): reason for node_id, reason in item.items()}
        for item in phase2.selection_exclusions
    ]
    pruned_nodes = dict(phase1.tree.pruned_nodes)
    pruned_nodes.update({mapping.get(node_id, node_id): reason for node_id, reason in phase2.tree.pruned_nodes.items()})
    stats = _combine_stats(dict(phase1.stats), dict(phase2.stats))
    stats["phase_two_triggered"] = True
    stats["phase_two_seed_node_id"] = phase1_best_node_id
    debug_stats = _combine_debug_stats(phase1.debug_stats, phase2.debug_stats, phase_two_mapping=mapping)
    leaderboard_history = [list(snapshot) for snapshot in phase1.tree.leaderboard_history] + phase2_history
    return _build_run_result(
        task,
        config,
        combined_nodes,
        combined_leaderboard,
        leaderboard_history,
        selection_history,
        selection_reasons,
        selection_exclusions,
        pruned_nodes,
        stats,
        debug_stats,
        workflow=workflow,
    )


def _run_search_stage(
    task: TaskSpec,
    config: StarkConfig,
    provider,
    evaluator,
    *,
    attempt_budget: int,
    deliberation_runner=None,
    initial_state: WorkflowInitialState | None = None,
    stage_state: StageRunState | None = None,
    stage_name: str = "phase",
) -> StageRunState:
    if stage_state is None:
        if initial_state is None:
            tree, root_eval, stats, debug_stats = bootstrap_stark_root(task, config, evaluator)
        else:
            tree, root_eval, stats, debug_stats = initial_state
        current_state = StageRunState(
            tree=tree,
            root_eval=root_eval,
            stats=stats,
            debug_stats=debug_stats,
            selection_history=[],
            selection_reasons=[],
            selection_exclusions=[],
            created_node_ids=[],
            feedback_state=collect_feedback_state(tree),
            deliberation_round=getattr(task.strategy_portfolio, "deliberation_round", 1) if task.strategy_portfolio else 1,
            last_upgrade_champion_id=None,
        )
    else:
        current_state = stage_state
    plan_agent = PlanAgent(provider)
    code_agent = CodeAgent(provider)
    debug_agent = DebugAgent(provider)
    if config.verbose:
        print(
            f"task={task.name} workflow=stark {stage_name} root status={current_state.tree.get_node(current_state.tree.root_id).status} "
            f"root runtime={shorten_runtime(current_state.root_eval.runtime)}",
            flush=True,
        )

    effective_attempt_index = 0
    transient_errors = 0
    transient_error_budget = _provider_transient_error_budget(attempt_budget)

    while effective_attempt_index < attempt_budget:
        current_state.feedback_state = collect_feedback_state(
            current_state.tree,
            plateau_delta_threshold=config.plateau_delta_threshold,
            plateau_window=config.plateau_window,
        )
        selected = current_state.tree.select_node(config)
        if selected is None:
            break
        selected_id, selection_reason = selected
        attempt_mode = _attempt_mode_for_selected_node(current_state.tree, selected_id, selection_reason, current_state.feedback_state)
        selected_node = current_state.tree.get_node(selected_id)
        displayed_attempt_index = effective_attempt_index + 1
        is_debug_attempt = selected_node.is_failure and selected_id != current_state.tree.root_id
        origin = "debug" if is_debug_attempt else "plan_code"

        if config.verbose:
            print(
                f"[{stage_name} attempt {displayed_attempt_index}] workflow=stark selected={selected_id} status={selected_node.status} "
                f"reason={selection_reason} mode={attempt_mode}",
                flush=True,
            )

        try:
            if is_debug_attempt:
                proposal, candidate_code, evaluation = _evaluate_debug(
                    task,
                    config,
                    evaluator,
                    debug_agent,
                    selected_node,
                    build_debug_context(current_state.tree, task, selected_id, config, current_state.feedback_state, attempt_mode),
                )
            else:
                proposal = plan_agent.run(
                    task,
                    selected_node,
                    build_plan_context(current_state.tree, task, selected_id, config, current_state.feedback_state, attempt_mode),
                )
                candidate_code, evaluation = _evaluate_plan_code(
                    task,
                    config,
                    evaluator,
                    code_agent,
                    selected_node,
                    proposal,
                    build_code_context(current_state.tree, task, selected_id, config, current_state.feedback_state, attempt_mode),
                    current_state.stats,
                )
        except Exception as exc:
            if not is_transient_provider_error(exc):
                raise
            transient_errors += 1
            _rollback_transient_selection(current_state.tree, selected_id)
            _record_provider_transient_error(
                current_state.stats,
                stage_name=stage_name,
                attempt_index=displayed_attempt_index,
                selected_id=selected_id,
                origin=origin,
                exc=exc,
                error_index=transient_errors,
                error_budget=transient_error_budget,
            )
            if config.verbose:
                print(
                    f"[{stage_name} attempt {displayed_attempt_index}] provider_transient_error "
                    f"{transient_errors}/{transient_error_budget}: {_short_error_message(exc)}",
                    flush=True,
                )
            if transient_errors >= transient_error_budget:
                current_state.stats["provider_transient_budget_exhausted_count"] = (
                    int(current_state.stats.get("provider_transient_budget_exhausted_count", 0)) + 1
                )
                if config.verbose:
                    print(
                        f"[{stage_name}] provider transient retry budget exhausted; finalizing current tree without consuming more attempts",
                        flush=True,
                    )
                break
            continue

        effective_attempt_index += 1
        _record_attempt_mode(current_state.stats, attempt_mode)
        current_state.selection_history.append(selected_id)
        current_state.selection_reasons.append(selection_reason)
        current_state.selection_exclusions.append(dict(current_state.tree.last_exclusions))
        current_state.stats["attempt_count"] += 1
        if is_debug_attempt:
            current_state.stats["debug_attempts"] += 1
            current_state.debug_stats["total_attempts"] += 1
            current_state.debug_stats["per_node"][selected_id] = current_state.debug_stats["per_node"].get(selected_id, 0) + 1
            selected_node.debug_attempts += 1
        else:
            current_state.stats["plan_attempts"] += 1

        child = current_state.tree.add_child(selected_id, candidate_code, proposal, evaluation, origin)
        current_state.created_node_ids.append(child.node_id)
        current_state.tree.update_leaderboard(child.node_id, config)
        current_state.tree.refresh_pruned_nodes(config)
        _record_failure(current_state.stats, evaluation)
        current_state.feedback_state = collect_feedback_state(current_state.tree)

        champion_id = getattr(current_state.feedback_state, "current_champion_id", None)
        if (
            config.deliberation_enabled
            and getattr(config, "deliberation_upgrade_enabled", True)
            and deliberation_runner is not None
            and task.strategy_portfolio is not None
            and getattr(current_state.feedback_state, "plateau_detected", False)
            and current_state.deliberation_round < getattr(config, "deliberation_max_rounds", 3)
            and (attempt_budget - effective_attempt_index) >= getattr(config, "deliberation_min_remaining_attempts", 8)
            and champion_id is not None
            and champion_id != current_state.last_upgrade_champion_id
        ):
            champion_summary, champion_code = _champion_upgrade_context(current_state.tree, current_state.feedback_state)
            if config.verbose:
                champion_speedup = getattr(current_state.feedback_state, "current_champion_speedup", None)
                champion_text = f"{champion_speedup:.3f}x" if isinstance(champion_speedup, (int, float)) else "n/a"
                print(
                    f"[deliberation_upgrade] stage={stage_name} attempt={effective_attempt_index} "
                    f"round={current_state.deliberation_round} champion={champion_text}",
                    flush=True,
                )
            upgraded_portfolio = deliberation_runner.run_upgrade(
                task,
                config,
                current_state.feedback_state,
                task.strategy_portfolio,
                current_state.deliberation_round + 1,
                champion_summary=champion_summary,
                champion_code=champion_code,
            )
            if upgraded_portfolio is not task.strategy_portfolio:
                task.strategy_portfolio = upgraded_portfolio
                current_state.deliberation_round = getattr(upgraded_portfolio, "deliberation_round", current_state.deliberation_round + 1)
                if config.verbose:
                    print(
                        f"[deliberation_upgrade] stage={stage_name} done round={current_state.deliberation_round} "
                        f"total_strategies={len(upgraded_portfolio.strategies)}",
                        flush=True,
                    )
            current_state.last_upgrade_champion_id = champion_id

        if config.verbose:
            print(
                f"  -> child={child.node_id} stage={stage_name} origin={origin} status={child.status} "
                f"score={shorten_runtime(child.runtime)} strategy={child.plan_strategy_name}",
                flush=True,
            )

    return current_state


def _phase_two_enabled(config: StarkConfig, deliberation_runner, task: TaskSpec) -> bool:
    if not bool(getattr(config, "phase_two_enabled", False)):
        return False
    if config.max_attempts <= 1:
        return False
    return True


def _phase_two_split(config: StarkConfig) -> int:
    requested = int(getattr(config, "phase_two_split_attempts", 5))
    return max(1, min(requested, max(1, config.max_attempts - 1)))


def _build_phase_two_task(
    task: TaskSpec,
    selected_node: SearchNode,
    selected_diagnostics,
    phase_transition: PhaseTransitionSummary,
) -> TaskSpec:
    return replace(
        task,
        source_code=selected_node.code,
        diagnostics_profile=selected_diagnostics,
        strategy_portfolio=None,
        phase_transition=phase_transition,
    )


def run_stark(
    task: TaskSpec,
    config: StarkConfig,
    provider,
    evaluator,
    deliberation_runner=None,
    initial_state: WorkflowInitialState | None = None,
) -> RunResult:
    """Run the main STARK search loop with tree memory and debug routing."""
    if not _phase_two_enabled(config, deliberation_runner, task):
        stage = _run_search_stage(
            task,
            config,
            provider,
            evaluator,
            attempt_budget=config.max_attempts,
            deliberation_runner=deliberation_runner,
            initial_state=initial_state,
            stage_name="phase1",
        )
        return _finalize_single_stage_run(task, config, stage, workflow="stark")

    phase1_budget = _phase_two_split(config)
    remaining_budget = max(0, config.max_attempts - phase1_budget)
    phase1 = _run_search_stage(
        task,
        config,
        provider,
        evaluator,
        attempt_budget=phase1_budget,
        deliberation_runner=deliberation_runner,
        initial_state=initial_state,
        stage_name="phase1",
    )
    if remaining_budget <= 0:
        return _finalize_single_stage_run(task, config, phase1, workflow="stark")

    phase1_best_node_id = _select_phase_two_seed(phase1.tree, phase1.created_node_ids)
    if phase1_best_node_id is None:
        continued = _run_search_stage(
            task,
            config,
            provider,
            evaluator,
            attempt_budget=remaining_budget,
            deliberation_runner=deliberation_runner,
            stage_state=phase1,
            stage_name="phase1b",
        )
        return _finalize_single_stage_run(task, config, continued, workflow="stark")

    phase_transition = _build_phase_transition_summary(task, config, phase1, phase1_best_node_id)
    selected_node = phase1.tree.get_node(phase1_best_node_id)
    selected_diagnostics = build_task_diagnostics(task, config, candidate_code=selected_node.code)
    phase_transition.selected_diagnostics = selected_diagnostics
    phase_transition.diagnostics_delta = _phase_diagnostics_delta(task.diagnostics_profile, selected_diagnostics)
    task.phase_transition = phase_transition

    phase2_task = _build_phase_two_task(task, selected_node, selected_diagnostics, phase_transition)
    phase2_task.strategy_portfolio = deliberation_runner.run(phase2_task, config) if deliberation_runner is not None else None
    phase2_initial_state = _initial_state_from_root_candidate(selected_node, config)
    phase2 = _run_search_stage(
        phase2_task,
        config,
        provider,
        evaluator,
        attempt_budget=remaining_budget,
        deliberation_runner=deliberation_runner,
        initial_state=phase2_initial_state,
        stage_name="phase2",
    )
    if phase2_task.strategy_portfolio is not None:
        task.strategy_portfolio = phase2_task.strategy_portfolio
    return _finalize_two_phase_run(
        task,
        config,
        phase1,
        phase2,
        phase1_best_node_id,
        workflow="stark",
    )
