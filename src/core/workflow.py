"""Main STARK workflow runner."""

from __future__ import annotations

from ..agents import CodeAgent, DebugAgent, PlanAgent
from ..feedback import collect_feedback_state
from .candidate import normalize_candidate
from .context import build_code_context, build_debug_context, build_plan_context, snapshot_node
from .static_check import check_candidate_static
from .regions import preserve_region_scaffold
from ..models import AgentContext, AnchorEdit, EvaluationResult, PlanProposal, RunResult, SearchNode, StarkConfig, TaskSpec
from .tree import TreeMemory
from ..utils import extract_anchor_names, preserve_anchor_scaffold, shorten_runtime


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
    }


def _new_debug_stats() -> dict:
    return {
        "total_attempts": 0,
        "per_node": {},
    }


WorkflowInitialState = tuple[TreeMemory, EvaluationResult, dict, dict]


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


def _finalize_run(
    task: TaskSpec,
    config: StarkConfig,
    tree: TreeMemory,
    stats: dict,
    debug_stats: dict,
    selection_history: list[str],
    selection_reasons: list[str],
    selection_exclusions: list[dict[str, str]],
    workflow: str,
    feedback_state,
) -> RunResult:
    best_node_id = tree.leaderboard[0] if tree.leaderboard else tree.root_id
    best_node = tree.get_node(best_node_id)
    stats["pruned_count"] = len(tree.pruned_nodes)
    feedback_state = collect_feedback_state(
        tree,
        plateau_delta_threshold=config.plateau_delta_threshold,
        plateau_window=config.plateau_window,
    )
    return RunResult(
        task_name=task.name,
        config=config,
        best_node_id=best_node_id,
        leaderboard=list(tree.leaderboard),
        nodes=tree.clone_nodes(),
        selection_history=selection_history,
        stats=stats,
        leaderboard_history=[list(snapshot) for snapshot in tree.leaderboard_history],
        selection_reasons=selection_reasons,
        selection_exclusions=selection_exclusions,
        pruned_nodes=dict(tree.pruned_nodes),
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
        feedback_state=feedback_state,
        reference_runtimes=dict(best_node.reference_runtimes),
        speedups=dict(best_node.speedups),
        primary_reference=best_node.primary_reference,
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
    if initial_state is None:
        tree, root_eval, stats, debug_stats = bootstrap_stark_root(task, config, evaluator)
    else:
        tree, root_eval, stats, debug_stats = initial_state
    feedback_state = collect_feedback_state(tree)
    plan_agent = PlanAgent(provider)
    code_agent = CodeAgent(provider)
    debug_agent = DebugAgent(provider)
    deliberation_round = getattr(task.strategy_portfolio, "deliberation_round", 1) if task.strategy_portfolio else 1
    last_upgrade_champion_id: str | None = None

    selection_history: list[str] = []
    selection_reasons: list[str] = []
    selection_exclusions: list[dict[str, str]] = []

    if config.verbose:
        print(
            f"task={task.name} workflow=stark root status={tree.get_node(tree.root_id).status} "
            f"root runtime={shorten_runtime(root_eval.runtime)}"
        )

    for attempt_index in range(1, config.max_attempts + 1):
        feedback_state = collect_feedback_state(
            tree,
            plateau_delta_threshold=config.plateau_delta_threshold,
            plateau_window=config.plateau_window,
        )
        selected = tree.select_node(config)
        if selected is None:
            break
        selected_id, selection_reason = selected
        attempt_mode = _attempt_mode_for_selected_node(tree, selected_id, selection_reason, feedback_state)
        _record_attempt_mode(stats, attempt_mode)
        selection_history.append(selected_id)
        selection_reasons.append(selection_reason)
        selection_exclusions.append(dict(tree.last_exclusions))
        selected_node = tree.get_node(selected_id)
        stats["attempt_count"] += 1

        if config.verbose:
            print(
                f"[attempt {attempt_index}] workflow=stark selected={selected_id} status={selected_node.status} "
                f"reason={selection_reason} mode={attempt_mode}"
            )

        if selected_node.is_failure and selected_id != tree.root_id:
            stats["debug_attempts"] += 1
            debug_stats["total_attempts"] += 1
            debug_stats["per_node"][selected_id] = debug_stats["per_node"].get(selected_id, 0) + 1
            selected_node.debug_attempts += 1
            proposal, candidate_code, evaluation = _evaluate_debug(
                task,
                config,
                evaluator,
                debug_agent,
                selected_node,
                build_debug_context(tree, task, selected_id, config, feedback_state, attempt_mode),
            )
            origin = "debug"
        else:
            stats["plan_attempts"] += 1
            proposal = plan_agent.run(task, selected_node, build_plan_context(tree, task, selected_id, config, feedback_state, attempt_mode))
            candidate_code, evaluation = _evaluate_plan_code(
                task,
                config,
                evaluator,
                code_agent,
                selected_node,
                proposal,
                build_code_context(tree, task, selected_id, config, feedback_state, attempt_mode),
                stats,
            )
            origin = "plan_code"

        child = tree.add_child(selected_id, candidate_code, proposal, evaluation, origin)
        tree.update_leaderboard(child.node_id, config)
        tree.refresh_pruned_nodes(config)
        _record_failure(stats, evaluation)
        feedback_state = collect_feedback_state(tree)
    
        champion_id = getattr(feedback_state, "current_champion_id", None)
        if (
            config.deliberation_enabled
            and getattr(config, "deliberation_upgrade_enabled", True)
            and deliberation_runner is not None
            and task.strategy_portfolio is not None
            and getattr(feedback_state, "plateau_detected", False)
            and deliberation_round < getattr(config, "deliberation_max_rounds", 3)
            and (config.max_attempts - attempt_index) >= getattr(config, "deliberation_min_remaining_attempts", 8)
            and champion_id is not None
            and champion_id != last_upgrade_champion_id
        ):
            champion_summary, champion_code = _champion_upgrade_context(tree, feedback_state)
            if config.verbose:
                champion_speedup = getattr(feedback_state, "current_champion_speedup", None)
                champion_text = f"{champion_speedup:.3f}x" if isinstance(champion_speedup, (int, float)) else "n/a"
                print(
                    f"[deliberation_upgrade] plateau detected at attempt={attempt_index} "
                    f"round={deliberation_round} champion={champion_text}",
                    flush=True,
                )
            upgraded_portfolio = deliberation_runner.run_upgrade(
                task,
                config,
                feedback_state,
                task.strategy_portfolio,
                deliberation_round + 1,
                champion_summary=champion_summary,
                champion_code=champion_code,
            )
            if upgraded_portfolio is not task.strategy_portfolio:
                task.strategy_portfolio = upgraded_portfolio
                deliberation_round = getattr(upgraded_portfolio, "deliberation_round", deliberation_round + 1)
                if config.verbose:
                    print(
                        f"[deliberation_upgrade] done round={deliberation_round} "
                        f"total_strategies={len(upgraded_portfolio.strategies)}",
                        flush=True,
                    )
            last_upgrade_champion_id = champion_id

        if config.verbose:
            print(
                f"  -> child={child.node_id} origin={origin} status={child.status} "
                f"score={shorten_runtime(child.runtime)} strategy={child.plan_strategy_name}"
            )

    return _finalize_run(
        task,
        config,
        tree,
        stats,
        debug_stats,
        selection_history,
        selection_reasons,
        selection_exclusions,
        workflow="stark",
        feedback_state=feedback_state,
    )


