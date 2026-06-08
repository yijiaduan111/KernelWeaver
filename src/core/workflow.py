"""Workflow runners for the STARK prototype.

This module hosts the main STARK workflow plus several paper-aligned
baselines and ablations:

- `run_stark`: multi-agent + grounded edits + dynamic context + search
- `run_sampling`: root-only sampling baseline
- `run_reflexion`: single-chain iterative refinement baseline
- `run_search_agent`: single-agent style search ablation
- `run_ma_only`: multi-agent best-of-K ablation without strategic search

All workflows produce the same `RunResult` shape so replay, summaries,
validation, and batch reporting stay compatible across modes.
"""

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


def _initialize_tree(task: TaskSpec, config: StarkConfig, evaluator) -> tuple[TreeMemory, EvaluationResult, dict, dict]:
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


def _record_attempt_mode(stats: dict, mode: str) -> None:
    counts = stats.setdefault("attempt_mode_counts", {})
    counts[mode] = counts.get(mode, 0) + 1


def _attempt_mode_for_index(attempt_index: int, config: StarkConfig, feedback_state) -> str:
    max_attempts = max(1, config.max_attempts)
    explore_cutoff = max(1, int(round(max_attempts * config.explore_fraction)))
    challenger_budget = max(1, int(round(max_attempts * config.challenger_fraction)))
    mutation_end = max_attempts - challenger_budget
    if attempt_index <= explore_cutoff:
        return "explore"
    if feedback_state is not None and getattr(feedback_state, "plateau_detected", False):
        if attempt_index < max_attempts:
            return "challenger"
    if attempt_index <= mutation_end:
        return "mutate_champion"
    if attempt_index < max_attempts:
        return "challenger"
    return "best_lineage_push"


def _select_node_for_mode(tree: TreeMemory, config: StarkConfig, attempt_mode: str, feedback_state) -> tuple[str, str] | None:
    if attempt_mode in {"mutate_champion", "best_lineage_push"} and feedback_state is not None:
        champion_id = getattr(feedback_state, "current_champion_id", None)
        if champion_id and champion_id in tree.nodes and tree.is_eligible(champion_id, config):
            node = tree.get_node(champion_id)
            node.selected_count += 1
            node.selection_reason = attempt_mode
            tree.last_exclusions = {}
            tree.selection_exclusion_history.append({})
            return champion_id, attempt_mode
    if attempt_mode == "challenger":
        eligible = tree.eligible_leaf_nodes(config)
        champion_lineage = set(getattr(getattr(feedback_state, "champion", None), "lineage", []) or [])
        challengers = [node_id for node_id in eligible if node_id not in champion_lineage]
        pool = challengers or eligible
        if pool:
            selected = min(
                pool,
                key=lambda node_id: (
                    tree.nodes[node_id].selected_count,
                    tree.nodes[node_id].depth,
                    node_id,
                ),
            )
            tree.nodes[selected].selected_count += 1
            tree.nodes[selected].selection_reason = "challenger"
            tree.last_exclusions = {}
            tree.selection_exclusion_history.append({})
            return selected, "challenger"
    selected = tree.select_node(config)
    if selected is None:
        return None
    selected_id, reason = selected
    if attempt_mode == "explore" and reason == "exploit_best_score":
        tree.get_node(selected_id).selection_reason = "explore"
        return selected_id, "explore"
    return selected_id, reason


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
    code_agent: CodeAgent,
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
    debug_agent: DebugAgent,
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
        strategy_portfolio=task.strategy_portfolio,
        feedback_state=feedback_state,
        reference_runtimes=dict(best_node.reference_runtimes),
        speedups=dict(best_node.speedups),
        primary_reference=best_node.primary_reference,
    )


def run_stark(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the main STARK search loop with tree memory and debug routing."""
    if config.verbose:
        print(f"[workflow] root_evaluation_start task={task.name}", flush=True)
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
    feedback_state = collect_feedback_state(tree)
    if config.verbose:
        print(
            f"[workflow] root_evaluation_done status={tree.get_node(tree.root_id).status} "
            f"runtime={shorten_runtime(root_eval.runtime)}",
            flush=True,
        )
    plan_agent = PlanAgent(provider)
    code_agent = CodeAgent(provider)
    debug_agent = DebugAgent(provider)

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
        attempt_mode = _attempt_mode_for_index(attempt_index, config, feedback_state)
        _record_attempt_mode(stats, attempt_mode)
        selected = _select_node_for_mode(tree, config, attempt_mode, feedback_state)
        if selected is None:
            break
        selected_id, selection_reason = selected
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


def run_sampling(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the sampling baseline.

    Each attempt starts from the root candidate, proposes one fresh
    grounded edit, and evaluates the result without using tree expansion
    for future selection or a separate debug branch.
    """
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
    feedback_state = collect_feedback_state(tree)
    root_node = tree.get_node(tree.root_id)
    plan_agent = PlanAgent(provider)
    code_agent = CodeAgent(provider)

    selection_history: list[str] = []
    selection_reasons: list[str] = []
    selection_exclusions: list[dict[str, str]] = []

    if config.verbose:
        print(
            f"task={task.name} workflow=sampling root status={root_node.status} "
            f"root runtime={shorten_runtime(root_eval.runtime)}"
        )

    for attempt_index in range(1, config.max_attempts + 1):
        root_node.selected_count += 1
        root_node.selection_reason = "sampling_root"
        selection_history.append(tree.root_id)
        selection_reasons.append("sampling_root")
        selection_exclusions.append({})
        stats["attempt_count"] += 1
        stats["plan_attempts"] += 1

        if config.verbose:
            print(f"[attempt {attempt_index}] workflow=sampling selected=root reason=sampling_root")

        proposal = plan_agent.run(task, root_node, _root_only_context(tree, "plan", feedback_state))
        candidate_code, evaluation = _evaluate_plan_code(
            task,
            config,
            evaluator,
            code_agent,
            root_node,
            proposal,
            _root_only_context(tree, "code", feedback_state),
            stats,
        )
        child = tree.add_child(tree.root_id, candidate_code, proposal, evaluation, "plan_code")
        tree.update_leaderboard(child.node_id, config)
        _record_failure(stats, evaluation)
        feedback_state = collect_feedback_state(tree)

        if config.verbose:
            print(
                f"  -> child={child.node_id} origin=plan_code status={child.status} "
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
        workflow="sampling",
        feedback_state=feedback_state,
    )


def run_search_agent(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the paper-style search ablation.

    This keeps strategic search and tree memory, but collapses generation
    into a single search-agent style step with no dedicated debug branch
    and no role-specific dynamic context window.
    """
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
    feedback_state = collect_feedback_state(tree)

    selection_history: list[str] = []
    selection_reasons: list[str] = []
    selection_exclusions: list[dict[str, str]] = []

    if config.verbose:
        print(
            f"task={task.name} workflow=search-agent root status={tree.get_node(tree.root_id).status} "
            f"root runtime={shorten_runtime(root_eval.runtime)}"
        )

    for attempt_index in range(1, config.max_attempts + 1):
        selected = tree.select_node(config)
        if selected is None:
            break
        selected_id, selection_reason = selected
        selection_history.append(selected_id)
        selection_reasons.append(selection_reason)
        selection_exclusions.append(dict(tree.last_exclusions))
        selected_node = tree.get_node(selected_id)
        stats["attempt_count"] += 1
        stats["plan_attempts"] += 1

        if config.verbose:
            print(
                f"[attempt {attempt_index}] workflow=search-agent selected={selected_id} "
                f"status={selected_node.status} reason={selection_reason}"
            )

        context = _search_only_context(tree, selected_id)
        proposal, candidate_code = provider.generate_search_candidate(task, selected_node, context)
        evaluation = evaluator.evaluate(task, candidate_code, config)
        child = tree.add_child(selected_id, candidate_code, proposal, evaluation, "plan_code")
        tree.update_leaderboard(child.node_id, config)
        tree.refresh_pruned_nodes(config)
        _record_failure(stats, evaluation)
        feedback_state = collect_feedback_state(tree)

        if config.verbose:
            print(
                f"  -> child={child.node_id} origin=plan_code status={child.status} "
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
        workflow="search-agent",
        feedback_state=feedback_state,
    )


def run_ma_only(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the paper-style multi-agent-only ablation.

    Each attempt starts from the root candidate, but the root sees the
    accumulated history of its children through the standard plan/code
    dynamic contexts. This preserves multi-agent coordination while
    removing strategic node selection.
    """
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
    feedback_state = collect_feedback_state(tree)
    root_node = tree.get_node(tree.root_id)
    plan_agent = PlanAgent(provider)
    code_agent = CodeAgent(provider)

    selection_history: list[str] = []
    selection_reasons: list[str] = []
    selection_exclusions: list[dict[str, str]] = []

    if config.verbose:
        print(
            f"task={task.name} workflow=ma-only root status={root_node.status} "
            f"root runtime={shorten_runtime(root_eval.runtime)}"
        )

    for attempt_index in range(1, config.max_attempts + 1):
        root_node.selected_count += 1
        root_node.selection_reason = "ma_only_root"
        selection_history.append(tree.root_id)
        selection_reasons.append("ma_only_root")
        selection_exclusions.append({})
        stats["attempt_count"] += 1
        stats["plan_attempts"] += 1

        if config.verbose:
            print(f"[attempt {attempt_index}] workflow=ma-only selected=root reason=ma_only_root")

        proposal = plan_agent.run(task, root_node, build_plan_context(tree, task, tree.root_id, config, feedback_state, "explore"))
        candidate_code, evaluation = _evaluate_plan_code(
            task,
            config,
            evaluator,
            code_agent,
            root_node,
            proposal,
            build_code_context(tree, task, tree.root_id, config, feedback_state, "explore"),
            stats,
        )
        child = tree.add_child(tree.root_id, candidate_code, proposal, evaluation, "plan_code")
        tree.update_leaderboard(child.node_id, config)
        _record_failure(stats, evaluation)
        feedback_state = collect_feedback_state(tree)

        if config.verbose:
            print(
                f"  -> child={child.node_id} origin=plan_code status={child.status} "
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
        workflow="ma-only",
        feedback_state=feedback_state,
    )


def run_reflexion(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the reflexion baseline.

    This keeps a single active branch. Successful nodes continue with a
    fresh plan/code step, while failing nodes are repaired through the
    debug path until the retry budget is exhausted.
    """
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
    feedback_state = collect_feedback_state(tree)
    plan_agent = PlanAgent(provider)
    code_agent = CodeAgent(provider)
    debug_agent = DebugAgent(provider)

    selection_history: list[str] = []
    selection_reasons: list[str] = []
    selection_exclusions: list[dict[str, str]] = []
    current_id = tree.root_id

    if config.verbose:
        print(
            f"task={task.name} workflow=reflexion root status={tree.get_node(tree.root_id).status} "
            f"root runtime={shorten_runtime(root_eval.runtime)}"
        )

    for attempt_index in range(1, config.max_attempts + 1):
        current = tree.get_node(current_id)
        if current.is_failure and current_id != tree.root_id and current.debug_attempts >= config.debug_retry_limit:
            stats["stopped_reason"] = "debug_retry_limit_reached"
            break

        current.selected_count += 1
        stats["attempt_count"] += 1
        selection_history.append(current_id)
        selection_exclusions.append({})

        if current.is_failure and current_id != tree.root_id:
            selection_reason = "reflexion_debug"
            current.selection_reason = selection_reason
            selection_reasons.append(selection_reason)
            stats["debug_attempts"] += 1
            debug_stats["total_attempts"] += 1
            debug_stats["per_node"][current_id] = debug_stats["per_node"].get(current_id, 0) + 1
            current.debug_attempts += 1
            proposal, candidate_code, evaluation = _evaluate_debug(
                task,
                config,
                evaluator,
                debug_agent,
                current,
                build_debug_context(tree, task, current_id, config, feedback_state, "mutate_champion"),
            )
            origin = "debug"
        else:
            selection_reason = "reflexion_chain"
            current.selection_reason = selection_reason
            selection_reasons.append(selection_reason)
            stats["plan_attempts"] += 1
            proposal = plan_agent.run(task, current, build_plan_context(tree, task, current_id, config, feedback_state, "explore"))
            candidate_code, evaluation = _evaluate_plan_code(
                task,
                config,
                evaluator,
                code_agent,
                current,
                proposal,
                build_code_context(tree, task, current_id, config, feedback_state, "explore"),
                stats,
            )
            origin = "plan_code"

        if config.verbose:
            print(
                f"[attempt {attempt_index}] workflow=reflexion selected={current_id} "
                f"status={current.status} reason={selection_reason}"
            )

        child = tree.add_child(current_id, candidate_code, proposal, evaluation, origin)
        tree.update_leaderboard(child.node_id, config)
        _record_failure(stats, evaluation)
        feedback_state = collect_feedback_state(tree)
        current_id = child.node_id

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
        workflow="reflexion",
        feedback_state=feedback_state,
    )


def run_workflow(task: TaskSpec, config: StarkConfig, provider, evaluator, workflow: str = "stark") -> RunResult:
    """Dispatch to one of the supported workflow modes."""
    if workflow == "stark":
        return run_stark(task, config, provider, evaluator)
    if workflow == "sampling":
        return run_sampling(task, config, provider, evaluator)
    if workflow == "reflexion":
        return run_reflexion(task, config, provider, evaluator)
    if workflow == "search-agent":
        return run_search_agent(task, config, provider, evaluator)
    if workflow == "ma-only":
        return run_ma_only(task, config, provider, evaluator)
    raise ValueError(f"Unsupported workflow: {workflow}")
