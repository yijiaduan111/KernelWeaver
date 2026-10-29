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
from .context import build_code_context, build_debug_context, build_plan_context, snapshot_node
from ..models import AgentContext, AnchorEdit, EvaluationResult, PlanProposal, RunResult, SearchNode, StarkConfig, TaskSpec
from .tree import TreeMemory
from .long_term_memory import TaskExperienceJournal
from .local_tuning import choose_local_variant
from .profile_feedback import profile_live_candidate
from ..utils import extract_anchor_names, normalize_anchor_scaffold, preserve_anchor_scaffold, shorten_runtime


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
    return preserve_anchor_scaffold(parent_code, candidate_code)


def _build_debug_proposal(node: SearchNode) -> PlanProposal:
    return PlanProposal(
        strategy_name=node.plan_strategy_name or "debug",
        strategy_summary=node.plan_summary or "Repair previous candidate",
        anchor_edits=list(node.anchor_edits),
        expected_gain="Recover a failing candidate with a local fix.",
        risk_notes="Debug route applies the smallest viable local repair.",
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
        "provider_errors": 0,
    }


def _new_debug_stats() -> dict:
    return {
        "total_attempts": 0,
        "per_node": {},
    }


def _evidence_workflow_guidance(task: TaskSpec, enabled: bool) -> dict:
    """Build a portable task contract inspired by KDA's public workflow.

    The contract deliberately contains no Blackwell-specific recipe.  Backend
    facts must be supplied by the evaluator or profiler for the current GPU.
    """
    if not enabled:
        return {}
    return {
        "method": "kda_style_evidence_loop",
        "objective": "Produce the fastest correct native kernel under the official evaluator.",
        "correctness_contract": [
            "Preserve the exact ModelNew interface and output semantics.",
            "Pass every official randomized correctness trial before performance promotion.",
            "Keep bounds, dtype, shape, layout, and int64 flattened-index constraints explicit.",
        ],
        "candidate_policy": [
            "Change one primary performance hypothesis per candidate.",
            "Ground the hypothesis in measured current/root/related-node evidence, retrieved lessons, or profiler evidence.",
            "Rank likely impact before cosmetic launch-parameter changes.",
            "Record failed hypotheses and do not repeat them without a concrete repair.",
        ],
        "promotion_rule": "Promote only a correct candidate whose measured runtime improves the relevant parent or root; never infer speed from code appearance.",
        "hardware_scope": f"Target the evaluator's current {task.backend or 'unknown'} device. Do not assume an architecture not observed in this run.",
    }


def _initialize_tree(task: TaskSpec, config: StarkConfig, evaluator) -> tuple[TreeMemory, EvaluationResult, dict, dict]:
    root_eval = evaluator.evaluate(task, task.source_code, config)
    tree = TreeMemory(_evaluation_to_root_node(task.source_code, root_eval), config)
    tree.update_leaderboard(tree.root_id, config)
    stats = _new_stats()
    debug_stats = _new_debug_stats()
    _record_failure(stats, root_eval)
    return tree, root_eval, stats, debug_stats


def _root_only_context(tree: TreeMemory, role: str) -> AgentContext:
    root = snapshot_node(tree, tree.root_id)
    return AgentContext(role=role, current=root, root=root, related=[], leaders=[], failure=None)


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
    candidate_code = code_agent.run(task, selected_node, proposal, code_context)
    normalized_code = normalize_anchor_scaffold(selected_node.code, candidate_code)
    if normalized_code is None:
        return candidate_code, _broken_anchor_evaluation(selected_node.code, candidate_code)
    return normalized_code, evaluator.evaluate(task, normalized_code, config)


def _evaluate_debug(
    task: TaskSpec,
    config: StarkConfig,
    evaluator,
    debug_agent: DebugAgent,
    selected_node: SearchNode,
    debug_context: AgentContext,
) -> tuple[PlanProposal, str, EvaluationResult]:
    candidate_code = debug_agent.run(task, selected_node, debug_context)
    proposal = _build_debug_proposal(selected_node)
    normalized_code = normalize_anchor_scaffold(selected_node.code, candidate_code)
    if normalized_code is None:
        return proposal, candidate_code, _broken_anchor_evaluation(selected_node.code, candidate_code)
    return proposal, normalized_code, evaluator.evaluate(task, normalized_code, config)


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
) -> RunResult:
    best_node_id = tree.leaderboard[0] if tree.leaderboard else tree.root_id
    best_node = tree.get_node(best_node_id)
    stats["pruned_count"] = len(tree.pruned_nodes)
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
        reference_runtimes=dict(best_node.reference_runtimes),
        speedups=dict(best_node.speedups),
        primary_reference=best_node.primary_reference,
    )


def run_stark(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the main STARK search loop with tree memory and debug routing."""
    # Retain old serialized fields for report loading, but fail closed if an
    # older profile tries to enable a removed knowledge channel.
    removed_fields = (
        "operator_memory_enabled", "operator_memory_public_only", "operator_memory_public_corpus",
        "operator_memory_path", "operator_memory_target_arch", "operator_memory_experimental_page_ids",
        "domain_wiki_enabled", "domain_wiki_path", "domain_wiki_target_arch", "domain_wiki_experimental_page_ids",
    )
    if any(getattr(config, name, None) for name in removed_fields):
        raise ValueError("Wiki is unavailable in the memory-only branch; use main_memory_longterm.")
    experience_journal = (TaskExperienceJournal(config.long_term_memory_path, config.long_term_memory_top_k)
                          if config.long_term_memory_enabled else None)
    task_experience = experience_journal.retrieve(task) if experience_journal else []
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
    plan_agent = PlanAgent(provider)
    code_agent = CodeAgent(provider)
    debug_agent = DebugAgent(provider)

    selection_history: list[str] = []
    selection_reasons: list[str] = []
    selection_exclusions: list[dict[str, str]] = []
    performance_feedback: dict = {}
    profile_attempted = False
    workflow_guidance = _evidence_workflow_guidance(task, config.evidence_workflow_enabled)
    stats["evidence_workflow_enabled"] = bool(workflow_guidance)
    stats["task_experience_count"] = len(task_experience)
    # Empty legacy report fields do not represent an enabled knowledge channel.
    stats["operator_memory_evidence"] = []
    stats["operator_memory_matches"] = []
    stats["domain_wiki_evidence"] = []
    stats["domain_wiki_matches"] = []
    if workflow_guidance:
        stats["task_contract"] = workflow_guidance

    if config.verbose:
        print(
            f"task={task.name} workflow=stark root status={tree.get_node(tree.root_id).status} "
            f"root runtime={shorten_runtime(root_eval.runtime)}"
        )

    for attempt_index in range(1, config.max_attempts + 1):
        tuning = None
        if (config.local_tuning_enabled and attempt_index >= config.local_tuning_start_attempt
                and stats.get("local_tune_attempts", 0) < config.local_tuning_max_attempts):
            tuning = choose_local_variant(tree, task.backend)
        if tuning is not None:
            parent, variant = tuning
            parent.selected_count += 1
            parent.selection_reason = "local_tune_validated_candidate"
            selected = (parent.node_id, parent.selection_reason)
            tree.last_exclusions = {}
        else:
            selected = tree.select_node(config)
        if selected is None:
            break
        selected_id, selection_reason = selected
        selection_history.append(selected_id)
        selection_reasons.append(selection_reason)
        selection_exclusions.append(dict(tree.last_exclusions))
        selected_node = tree.get_node(selected_id)
        stats["attempt_count"] += 1

        # Task experience is advisory: reserve deterministic attempts for
        # fresh exploration so historical CUDA strategies cannot suppress new
        # candidates. The ratio is rounded to an integer number of attempts.
        exploration_ratio = min(1.0, max(0.0, float(config.long_term_memory_exploration_ratio)))
        exploration_slots = int(round(config.max_attempts * exploration_ratio))
        use_long_term = bool(task_experience) and (attempt_index > exploration_slots)
        active_long_term = task_experience if use_long_term else []

        if config.verbose:
            print(
                f"[attempt {attempt_index}] workflow=stark selected={selected_id} status={selected_node.status} "
                f"reason={selection_reason}"
            )

        try:
            if tuning is not None:
                stats["local_tune_attempts"] = stats.get("local_tune_attempts", 0) + 1
                proposal = variant.proposal()
                candidate_code = variant.code
                evaluation = evaluator.evaluate(task, candidate_code, config)
                origin = "local_tune"
                stats.setdefault("local_tuning_trials", []).append({
                    "attempt": attempt_index, "parent_id": selected_id,
                    **variant.metadata(), "correct": evaluation.correct,
                    "compile_ok": evaluation.compile_ok, "runtime": evaluation.runtime,
                    "parent_runtime": selected_node.runtime,
                })
            elif selected_node.is_failure and selected_id != tree.root_id:
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
                    build_debug_context(tree, selected_id, config, active_long_term),
                )
                origin = "debug"
            else:
                stats["plan_attempts"] += 1
                proposal = plan_agent.run(
                    task, selected_node,
                    build_plan_context(
                        tree, selected_id, config, active_long_term,
                        performance_feedback, workflow_guidance,
                    ),
                )
                candidate_code, evaluation = _evaluate_plan_code(
                    task,
                    config,
                    evaluator,
                    code_agent,
                    selected_node,
                    proposal,
                    build_code_context(tree, selected_id, config, active_long_term),
                    stats,
                )
                origin = "plan_code"
        except RuntimeError as exc:
            if "llm " not in str(exc).lower():
                raise
            stats["provider_errors"] = stats.get("provider_errors", 0) + 1
            if config.verbose:
                print(f"  -> provider_error attempt={attempt_index} error={exc}")
            continue

        child = tree.add_child(selected_id, candidate_code, proposal, evaluation, origin)
        tree.update_leaderboard(child.node_id, config)
        tree.refresh_pruned_nodes(config)
        _record_failure(stats, evaluation)
        if (config.profile_feedback_enabled and not profile_attempted
                and task.backend == "cuda" and child.correct
                and attempt_index < config.max_attempts
                and child.speedup is not None and child.speedup >= 0.7):
            profile_attempted = True
            performance_feedback = profile_live_candidate(task, config, child, stats)

        if config.verbose:
            print(
                f"  -> child={child.node_id} origin={origin} status={child.status} "
                f"score={shorten_runtime(child.runtime)} strategy={child.plan_strategy_name}"
            )

    result = _finalize_run(
        task,
        config,
        tree,
        stats,
        debug_stats,
        selection_history,
        selection_reasons,
        selection_exclusions,
        workflow="stark",
    )
    if experience_journal:
        experience_journal.record(task, result)
    return result


def run_sampling(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the sampling baseline.

    Each attempt starts from the root candidate, proposes one fresh
    grounded edit, and evaluates the result without using tree expansion
    for future selection or a separate debug branch.
    """
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
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

        proposal = plan_agent.run(task, root_node, _root_only_context(tree, "plan"))
        candidate_code, evaluation = _evaluate_plan_code(
            task,
            config,
            evaluator,
            code_agent,
            root_node,
            proposal,
            _root_only_context(tree, "code"),
            stats,
        )
        child = tree.add_child(tree.root_id, candidate_code, proposal, evaluation, "plan_code")
        tree.update_leaderboard(child.node_id, config)
        _record_failure(stats, evaluation)

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
    )


def run_search_agent(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the paper-style search ablation.

    This keeps strategic search and tree memory, but collapses generation
    into a single search-agent style step with no dedicated debug branch
    and no role-specific dynamic context window.
    """
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)

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
    )


def run_ma_only(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the paper-style multi-agent-only ablation.

    Each attempt starts from the root candidate, but the root sees the
    accumulated history of its children through the standard plan/code
    dynamic contexts. This preserves multi-agent coordination while
    removing strategic node selection.
    """
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
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

        proposal = plan_agent.run(task, root_node, build_plan_context(tree, tree.root_id, config))
        candidate_code, evaluation = _evaluate_plan_code(
            task,
            config,
            evaluator,
            code_agent,
            root_node,
            proposal,
            build_code_context(tree, tree.root_id, config),
            stats,
        )
        child = tree.add_child(tree.root_id, candidate_code, proposal, evaluation, "plan_code")
        tree.update_leaderboard(child.node_id, config)
        _record_failure(stats, evaluation)

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
    )


def run_reflexion(task: TaskSpec, config: StarkConfig, provider, evaluator) -> RunResult:
    """Run the reflexion baseline.

    This keeps a single active branch. Successful nodes continue with a
    fresh plan/code step, while failing nodes are repaired through the
    debug path until the retry budget is exhausted.
    """
    tree, root_eval, stats, debug_stats = _initialize_tree(task, config, evaluator)
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
                build_debug_context(tree, current_id, config),
            )
            origin = "debug"
        else:
            selection_reason = "reflexion_chain"
            current.selection_reason = selection_reason
            selection_reasons.append(selection_reason)
            stats["plan_attempts"] += 1
            proposal = plan_agent.run(task, current, build_plan_context(tree, current_id, config))
            candidate_code, evaluation = _evaluate_plan_code(
                task,
                config,
                evaluator,
                code_agent,
                current,
                proposal,
                build_code_context(tree, current_id, config),
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
