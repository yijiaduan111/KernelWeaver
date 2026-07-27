"""Runner for the direct one-shot LLM KernelBench baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..models import EvaluationResult, RunResult, SearchNode, StarkConfig, TaskSpec
from .code_extract import extract_python_code
from .prompts import build_direct_system_prompt, build_direct_user_payload


DIRECT_WORKFLOW = "direct_llm_baseline"
CANDIDATE_NODE_ID = "candidate_1"


def run_direct_baseline(
    task: TaskSpec,
    config: StarkConfig,
    provider,
    evaluator,
    *,
    artifact_dir: str | Path | None = None,
    temperature: float = 0.2,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> RunResult:
    """Generate and evaluate exactly one direct LLM candidate.

    Provider transport retries are handled below the model-call boundary. Once a
    text response is received, this runner performs no repair, no debug loop,
    and no second model attempt.
    """
    started_at = time.time()
    artifact_path = Path(artifact_dir) if artifact_dir is not None else None
    if artifact_path is not None:
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / "reference_source.py").write_text(task.reference_code, encoding="utf-8")

    backend = task.backend or "cuda"
    raw_response = provider.generate_text(
        build_direct_system_prompt(backend),
        build_direct_user_payload(task, backend),
        temperature=temperature,
        purpose="direct_kernelbench_baseline",
    )
    candidate_code = extract_python_code(raw_response)

    if artifact_path is not None:
        (artifact_path / "raw_response.txt").write_text(raw_response, encoding="utf-8")
        (artifact_path / "candidate.py").write_text(candidate_code, encoding="utf-8")

    if candidate_code.strip():
        candidate_eval = evaluator.evaluate(task, candidate_code, config)
    else:
        candidate_eval = _failure_result(
            "compile",
            "empty_direct_response",
            "direct_baseline_error: provider returned no extractable Python source",
        )

    elapsed_seconds = time.time() - started_at
    result = _build_direct_run_result(
        task=task,
        config=config,
        candidate_code=candidate_code,
        candidate_eval=candidate_eval,
        elapsed_seconds=elapsed_seconds,
        raw_response_chars=len(raw_response or ""),
        provider_name=provider_name or getattr(provider, "name", None),
        model_name=model_name,
        temperature=temperature,
    )

    if artifact_path is not None:
        metadata = {
            "workflow": DIRECT_WORKFLOW,
            "provider": provider_name or getattr(provider, "name", None),
            "model": model_name,
            "temperature": temperature,
            "elapsed_seconds": elapsed_seconds,
            "raw_response_chars": len(raw_response or ""),
            "candidate_code_chars": len(candidate_code or ""),
            "candidate_compile_ok": candidate_eval.compile_ok,
            "candidate_correct": candidate_eval.correct,
            "candidate_speedup": candidate_eval.speedup,
            "failure_stage": candidate_eval.failure_stage,
            "failure_type": candidate_eval.failure_type,
        }
        (artifact_path / "direct_baseline.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return result


def _build_direct_run_result(
    *,
    task: TaskSpec,
    config: StarkConfig,
    candidate_code: str,
    candidate_eval: EvaluationResult,
    elapsed_seconds: float,
    raw_response_chars: int,
    provider_name: str | None,
    model_name: str | None,
    temperature: float,
) -> RunResult:
    root_node = SearchNode(
        node_id="root",
        parent_id=None,
        depth=0,
        code=task.reference_code,
        origin="official_reference_not_evaluated",
        compile_ok=False,
        correct=False,
        runtime=None,
        score=float("inf"),
        logs=["direct_baseline_root_not_evaluated=true"],
        node_status="not_evaluated",
        latest_failure_stage=None,
    )
    candidate_node = _candidate_node(candidate_code, candidate_eval)
    root_node.child_ids.append(candidate_node.node_id)
    nodes = {root_node.node_id: root_node, candidate_node.node_id: candidate_node}
    leaderboard = [candidate_node.node_id] if candidate_node.compile_ok and candidate_node.correct else []
    stats = _direct_stats(
        candidate_eval,
        elapsed_seconds=elapsed_seconds,
        raw_response_chars=raw_response_chars,
        candidate_code_chars=len(candidate_code or ""),
        provider_name=provider_name,
        model_name=model_name,
        temperature=temperature,
    )
    return RunResult(
        task_name=task.name,
        config=config,
        best_node_id=candidate_node.node_id,
        leaderboard=leaderboard,
        nodes=nodes,
        selection_history=["root"],
        stats=stats,
        leaderboard_history=[leaderboard],
        selection_reasons=["direct_llm_one_shot"],
        selection_exclusions=[{}],
        pruned_nodes={},
        debug_stats={"total_attempts": 0, "per_node": {}},
        benchmark_family=task.benchmark_family,
        level=task.level,
        problem_id=task.problem_id,
        backend=task.backend,
        source_origin=task.source_origin,
        source_root=task.source_root,
        workflow=DIRECT_WORKFLOW,
        run_profile=config.run_profile,
        search_profile=config.search_profile,
        evaluator_profile=config.evaluator_profile,
        measurement_profile=config.measurement_profile,
        preset=config.preset,
        evaluation_profile=config.evaluation_profile,
        kernelbench_evaluator=config.kernelbench_evaluator,
        grounded_regions=[],
        semantic_profile=None,
        diagnostics_profile=None,
        strategy_portfolio=None,
        phase_transition=None,
        feedback_state=None,
        reference_runtimes=dict(candidate_node.reference_runtimes),
        speedups=dict(candidate_node.speedups),
        primary_reference=candidate_node.primary_reference,
    )


def _candidate_node(code: str, evaluation: EvaluationResult) -> SearchNode:
    return SearchNode(
        node_id=CANDIDATE_NODE_ID,
        parent_id="root",
        depth=1,
        code=code,
        origin="direct_llm_generation",
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
        plan_strategy_name="direct_llm_one_shot",
        plan_summary="Single direct model generation without STARK planning, search, deliberation, or debug repair.",
        plan_mode="explore",
        target_metric="speedup",
    )


def _direct_stats(
    evaluation: EvaluationResult,
    *,
    elapsed_seconds: float,
    raw_response_chars: int,
    candidate_code_chars: int,
    provider_name: str | None,
    model_name: str | None,
    temperature: float,
) -> dict[str, Any]:
    failure_counts: dict[str, int] = {}
    failure_stage_counts: dict[str, int] = {}
    if evaluation.failure_type:
        failure_counts[evaluation.failure_type] = 1
    if evaluation.failure_stage and evaluation.failure_stage != "none":
        failure_stage_counts[evaluation.failure_stage] = 1
    return {
        "attempt_count": 1,
        "model_call_count": 1,
        "direct_baseline": True,
        "provider": provider_name,
        "model": model_name,
        "temperature": temperature,
        "elapsed_seconds": elapsed_seconds,
        "raw_response_chars": raw_response_chars,
        "candidate_code_chars": candidate_code_chars,
        "candidate_compile_count": int(evaluation.compile_ok),
        "candidate_correct_count": int(evaluation.correct),
        "failure_counts": failure_counts,
        "failure_stage_counts": failure_stage_counts,
        "pruned_count": 0,
    }


def _node_status_from_evaluation(evaluation: EvaluationResult) -> str:
    if evaluation.failure_stage == "compile":
        return "compile_fail"
    if evaluation.failure_stage == "runtime":
        return "runtime_fail"
    if evaluation.failure_stage == "correctness":
        return "correctness_fail"
    return "correct"


def _failure_result(stage: str, failure_type: str, *logs: str) -> EvaluationResult:
    return EvaluationResult(
        compile_ok=stage != "compile",
        correct=False,
        runtime=None,
        score=float("inf"),
        logs=list(logs),
        failure_type=failure_type,
        failure_stage=stage,
        reference_runtime=None,
        speedup=None,
        reference_runtimes={},
        speedups={},
        primary_reference="torch_eager",
    )
