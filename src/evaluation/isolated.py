from __future__ import annotations

"""Subprocess isolation for KernelBench candidate evaluation.

Fatal CUDA errors can poison the current Python process. This wrapper evaluates
one candidate in a short-lived worker so later candidates start from a clean CUDA
context while LLM providers remain loaded in the parent process.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..models import EvaluationResult, StarkConfig, TaskSpec
from .base import Evaluator, _failure_result


class IsolatedEvaluator(Evaluator):
    """Run each evaluator call in a fresh Python subprocess."""

    def __init__(self, inner: Evaluator, timeout_seconds: int = 900):
        self.inner = inner
        self.timeout_seconds = int(timeout_seconds or 900)
        self.repo_root = Path(__file__).resolve().parents[2]

    def evaluate(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        if getattr(config, "evaluator_isolation", "off") != "candidate_subprocess":
            return self.inner.evaluate(task, code, config)
        payload = {
            "task": _task_payload(task),
            "config": asdict(config),
            "code": code,
        }
        with tempfile.TemporaryDirectory(prefix="kw_eval_") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.json"
            output_path = temp_path / "output.json"
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "src.evaluation.worker",
                str(input_path),
                str(output_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.repo_root),
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                return _isolated_failure(
                    "isolated_evaluator_timeout",
                    [
                        f"isolated_worker_timed_out=true",
                        f"isolated_worker_timeout_seconds={self.timeout_seconds}",
                        _tail_log("isolated_worker_stdout_tail", exc.stdout),
                        _tail_log("isolated_worker_stderr_tail", exc.stderr),
                    ],
                )
            if completed.returncode != 0:
                return _isolated_failure(
                    "isolated_evaluator_crash",
                    [
                        f"isolated_worker_exit_code={completed.returncode}",
                        _tail_log("isolated_worker_stdout_tail", completed.stdout),
                        _tail_log("isolated_worker_stderr_tail", completed.stderr),
                    ],
                )
            if not output_path.exists():
                return _isolated_failure(
                    "isolated_evaluator_missing_output",
                    [
                        "isolated_worker_output_missing=true",
                        _tail_log("isolated_worker_stdout_tail", completed.stdout),
                        _tail_log("isolated_worker_stderr_tail", completed.stderr),
                    ],
                )
            try:
                data = json.loads(output_path.read_text(encoding="utf-8"))
                result = _evaluation_from_payload(data)
            except Exception as exc:
                return _isolated_failure(
                    "isolated_evaluator_bad_output",
                    [
                        f"isolated_worker_bad_output={exc}",
                        _tail_log("isolated_worker_stdout_tail", completed.stdout),
                        _tail_log("isolated_worker_stderr_tail", completed.stderr),
                    ],
                )
            result.logs.append("isolated_evaluator=candidate_subprocess")
            if completed.stderr.strip():
                result.logs.append(_tail_log("isolated_worker_stderr_tail", completed.stderr))
            return result


def _task_payload(task: TaskSpec) -> dict[str, Any]:
    return {
        "name": task.name,
        "description": task.description,
        "source_code": task.source_code,
        "reference_code": task.reference_code,
        "function_name": task.function_name,
        "reference_function_name": task.reference_function_name,
        "tags": list(task.tags),
        "source_origin": task.source_origin,
        "benchmark_family": task.benchmark_family,
        "entry_kind": task.entry_kind,
        "level": task.level,
        "problem_id": task.problem_id,
        "backend": task.backend,
        "source_root": task.source_root,
    }


def _evaluation_from_payload(payload: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        compile_ok=bool(payload.get("compile_ok", False)),
        correct=bool(payload.get("correct", False)),
        runtime=payload.get("runtime"),
        score=float(payload.get("score", float("inf"))),
        logs=list(payload.get("logs") or []),
        failure_type=payload.get("failure_type"),
        failure_stage=str(payload.get("failure_stage", "none")),
        reference_runtime=payload.get("reference_runtime"),
        speedup=payload.get("speedup"),
        reference_runtimes=dict(payload.get("reference_runtimes") or {}),
        speedups=dict(payload.get("speedups") or {}),
        primary_reference=payload.get("primary_reference"),
    )


def _isolated_failure(failure_type: str, logs: list[str]) -> EvaluationResult:
    return _failure_result("runtime", failure_type, *[log for log in logs if log])


def _tail_log(label: str, value: Any, limit: int = 2000) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if len(text) > limit:
        text = text[-limit:]
    return f"{label}={text}"
