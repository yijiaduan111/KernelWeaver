"""Summary-row helpers for direct baseline batch runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..experiment import candidate_attempt_stats, runtime_for_mode, speedup_for_mode


def direct_result_row(
    result,
    *,
    alias: str,
    level: int,
    problem_id: int,
    backend: str,
    run_path: str | Path | None,
    error: str | None = None,
) -> dict[str, Any]:
    best = result.nodes[result.best_node_id]
    stats = candidate_attempt_stats(result)
    return {
        "alias": alias,
        "level": level,
        "problem_id": problem_id,
        "backend": backend,
        "workflow": result.workflow,
        "run_profile": result.run_profile,
        "search_profile": result.search_profile,
        "evaluator_profile": result.evaluator_profile,
        "measurement_profile": result.measurement_profile,
        "preset": result.preset,
        "evaluation_profile": result.evaluation_profile,
        "kernelbench_evaluator": result.kernelbench_evaluator,
        "status": "ok" if error is None else "error",
        "task_name": result.task_name,
        "best_node_id": result.best_node_id,
        "best_status": best.status,
        "best_node_is_root": result.best_node_id == "root",
        "best_correct": bool(best.compile_ok and best.correct),
        "paper_fast1": bool(best.correct and isinstance(best.speedup, (int, float)) and best.speedup >= 1.0),
        "root_correct": bool(result.nodes.get("root") and result.nodes["root"].correct),
        "non_root_correct": any(node.correct and node_id != "root" for node_id, node in result.nodes.items()),
        "improved_over_reference": bool(best.runtime is not None and best.reference_runtime is not None and best.runtime < best.reference_runtime),
        "candidate_runtime": best.runtime,
        "reference_runtime": best.reference_runtime,
        "speedup": best.speedup,
        "primary_reference": best.primary_reference or result.primary_reference or "torch_eager",
        "torch_eager_reference_runtime": runtime_for_mode(best.reference_runtimes, "torch_eager", best.reference_runtime),
        "torch_compile_default_reference_runtime": runtime_for_mode(best.reference_runtimes, "torch_compile_default"),
        "torch_compile_max_autotune_reference_runtime": runtime_for_mode(best.reference_runtimes, "torch_compile_max_autotune"),
        "torch_eager_speedup": speedup_for_mode(best.speedups, "torch_eager", best.speedup),
        "torch_compile_default_speedup": speedup_for_mode(best.speedups, "torch_compile_default"),
        "torch_compile_max_autotune_speedup": speedup_for_mode(best.speedups, "torch_compile_max_autotune"),
        "candidate_total_count": stats["total"],
        "candidate_compile_count": stats["compile"],
        "candidate_correct_count": stats["correct"],
        "compile_rate": stats["compile_rate"],
        "correct_rate": stats["correct_rate"],
        "failure_stage": best.latest_failure_stage or "none",
        "failure_type": best.failure_type,
        "validation_correctness_matches": None,
        "validation_speed_direction_matches": None,
        "run_path": str(run_path) if run_path is not None else None,
        "validation_path": None,
        "error": error,
    }
