"""Shadow validation helpers for saved KernelBench runs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..core.bridge import KernelBenchTaskBridge
from ..io import load_run
from .evaluator_paper import KernelBenchPaperEvaluator


def verify_kernelbench_run(run_path: str | Path, kernelbench_root: str | Path | None = None, output_path: str | Path | None = None):
    run_file = Path(run_path)
    result = load_run(run_file)
    if result.benchmark_family != "kernelbench":
        raise ValueError(f"Only KernelBench runs can be verified, got benchmark_family={result.benchmark_family!r}")
    if result.level is None or result.problem_id is None:
        raise ValueError("KernelBench verification requires level and problem_id metadata.")
    resolved_root = str(kernelbench_root or result.source_root or "")
    if not resolved_root:
        raise ValueError("KernelBench verification requires kernelbench_root or a recorded source_root.")
    best = result.nodes[result.best_node_id]
    best_code_path = run_file.parent / "best_code.py"
    if not best_code_path.exists():
        raise FileNotFoundError(f"Missing best_code.py next to run.json: {best_code_path}")

    evaluator_kind = result.kernelbench_evaluator or getattr(result.config, "kernelbench_evaluator", "paper")
    if evaluator_kind != "paper":
        raise ValueError("Legacy local KernelBench evaluator has been removed. Only paper-based verification is supported now.")

    backend = result.backend or "triton"
    task = KernelBenchTaskBridge().load_official_problem(resolved_root, result.level, result.problem_id, backend=backend)
    validation_config = replace(
        result.config,
        benchmark_loops=max(120, result.config.benchmark_loops),
        warmup_loops=max(10, result.config.warmup_loops),
        num_correct_trials=max(3, result.config.num_correct_trials),
        num_perf_trials=max(30, result.config.num_perf_trials),
        paper_num_warmup=max(5, result.config.paper_num_warmup),
        kernelbench_evaluator="paper",
    )
    candidate_code = best_code_path.read_text(encoding="utf-8")
    validation_result = KernelBenchPaperEvaluator().evaluate(task, candidate_code, validation_config)

    payload = {
        "task_name": result.task_name,
        "benchmark_family": result.benchmark_family,
        "level": result.level,
        "problem_id": result.problem_id,
        "backend": result.backend,
        "run_profile": result.run_profile,
        "search_profile": result.search_profile,
        "evaluator_profile": result.evaluator_profile,
        "measurement_profile": result.measurement_profile,
        "kernelbench_evaluator": "paper",
        "best_node_id": result.best_node_id,
        "kernelbench_root": resolved_root,
        "run_path": str(run_file),
        "best_code_path": str(best_code_path),
        "recorded": {
            "correct": best.correct,
            "candidate_runtime": best.runtime,
            "reference_runtime": best.reference_runtime,
            "speedup": best.speedup,
            "reference_runtimes": dict(best.reference_runtimes),
            "speedups": dict(best.speedups),
            "primary_reference": best.primary_reference or result.primary_reference,
            "speed_direction": _speed_direction(best.runtime, best.reference_runtime),
            "failure_stage": best.latest_failure_stage or "none",
            "failure_type": best.failure_type,
        },
        "validation": {
            "compile_ok": validation_result.compile_ok,
            "correct": validation_result.correct,
            "candidate_runtime": validation_result.runtime,
            "reference_runtime": validation_result.reference_runtime,
            "speedup": validation_result.speedup,
            "reference_runtimes": dict(validation_result.reference_runtimes),
            "speedups": dict(validation_result.speedups),
            "primary_reference": validation_result.primary_reference,
            "speed_direction": _speed_direction(validation_result.runtime, validation_result.reference_runtime),
            "failure_stage": validation_result.failure_stage,
            "failure_type": validation_result.failure_type,
            "logs": list(validation_result.logs),
        },
        "checks": {
            "correctness_matches": best.correct == validation_result.correct,
            "speed_direction_matches": _speed_direction(best.runtime, best.reference_runtime)
            == _speed_direction(validation_result.runtime, validation_result.reference_runtime),
        },
    }
    target = Path(output_path) if output_path is not None else run_file.parent / "validation.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def load_validation(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if candidate.name == "run.json":
        candidate = candidate.parent / "validation.json"
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _speed_direction(candidate_runtime: float | None, reference_runtime: float | None) -> str:
    if candidate_runtime is None or reference_runtime is None:
        return "unknown"
    if reference_runtime <= 0 or candidate_runtime <= 0:
        return "unknown"
    ratio = candidate_runtime / reference_runtime
    if 0.88 <= ratio <= 1.12:
        return "equal"
    return "faster" if ratio < 1.0 else "slower"
