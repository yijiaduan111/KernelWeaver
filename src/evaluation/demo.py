from __future__ import annotations

import time
from typing import Any

from ..models import EvaluationResult, StarkConfig, TaskSpec
from ..utils import compare_values
from .base import Evaluator, _clone_args, _clone_kwargs, _failure_result, _load_callable_from_source, _require_torch, _success_result


class DemoEvaluator(Evaluator):
    def __init__(self) -> None:
        self._reference_cache: dict[str, Any] = {}

    def evaluate(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        if task.entry_kind != "callable":
            return _failure_result("compile", "compile_error", f"DemoEvaluator only supports callable tasks, got entry_kind={task.entry_kind}")
        try:
            candidate_fn = self._load_function(code, task.function_name)
        except Exception as exc:
            return _failure_result("compile", "compile_error", f"compile_error: {exc}")
        try:
            reference_fn = self._load_reference(task)
        except Exception as exc:
            return _failure_result("compile", "reference_error", f"reference_error: {exc}")
        for case in task.test_cases:
            try:
                actual = candidate_fn(*_clone_args(case.args), **_clone_kwargs(case.kwargs))
            except Exception as exc:
                return _failure_result("runtime", "runtime_error", f"runtime_error[{case.label}]: {exc}")
            try:
                expected = reference_fn(*_clone_args(case.args), **_clone_kwargs(case.kwargs))
            except Exception as exc:
                return _failure_result("compile", "reference_error", f"reference_error[{case.label}]: {exc}")
            if not compare_values(actual, expected):
                return _failure_result("correctness", "correctness_error", f"correctness_error[{case.label}]")
        try:
            runtime = self._measure_runtime(candidate_fn, task, config)
        except Exception as exc:
            return _failure_result("runtime", "runtime_error", f"runtime_error[benchmark]: {exc}")
        return _success_result(runtime)

    def _load_reference(self, task: TaskSpec):
        if task.name not in self._reference_cache:
            self._reference_cache[task.name] = self._load_function(task.reference_code, task.reference_function_name)
        return self._reference_cache[task.name]

    @staticmethod
    def _load_function(source_code: str, function_name: str):
        namespace: dict[str, object] = {}
        exec(source_code, namespace)
        if function_name not in namespace or not callable(namespace[function_name]):
            raise ValueError(f"Entrypoint '{function_name}' is missing.")
        return namespace[function_name]

    @staticmethod
    def _measure_runtime(candidate_fn, task: TaskSpec, config: StarkConfig) -> float:
        for _ in range(config.warmup_loops):
            for case in task.benchmark_cases:
                candidate_fn(*case.args, **case.kwargs)
        started = time.perf_counter()
        for _ in range(config.benchmark_loops):
            for case in task.benchmark_cases:
                candidate_fn(*case.args, **case.kwargs)
        elapsed = time.perf_counter() - started
        return elapsed / (config.benchmark_loops * max(len(task.benchmark_cases), 1))


class TritonEvaluator(Evaluator):
    def evaluate(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        if task.entry_kind != "callable":
            return _failure_result("compile", "compile_error", f"TritonEvaluator only supports callable tasks, got entry_kind={task.entry_kind}")
        torch = _require_torch()
        try:
            with _load_callable_from_source(code, task.function_name, prefix=f"candidate_{task.name}") as candidate_fn:
                with _load_callable_from_source(task.reference_code, task.reference_function_name, prefix=f"reference_{task.name}") as reference_fn:
                    for case in task.test_cases:
                        try:
                            actual = candidate_fn(*_clone_args(case.args), **_clone_kwargs(case.kwargs))
                            expected = reference_fn(*_clone_args(case.args), **_clone_kwargs(case.kwargs))
                            torch.cuda.synchronize()
                        except Exception as exc:
                            return _failure_result("runtime", "runtime_error", f"runtime_error[{case.label}]: {exc}")
                        if not compare_values(actual, expected, tolerance=1e-4):
                            return _failure_result("correctness", "correctness_error", f"correctness_error[{case.label}]")
                    try:
                        runtime = self._measure_runtime(candidate_fn, task, config)
                    except Exception as exc:
                        return _failure_result("runtime", "runtime_error", f"runtime_error[benchmark]: {exc}")
        except Exception as exc:
            return _failure_result("compile", "compile_error", f"compile_error: {exc}")
        return _success_result(runtime)

    @staticmethod
    def _measure_runtime(candidate_fn, task: TaskSpec, config: StarkConfig) -> float:
        torch = _require_torch()
        for _ in range(config.warmup_loops):
            for case in task.benchmark_cases:
                candidate_fn(*case.args, **case.kwargs)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(config.benchmark_loops):
            for case in task.benchmark_cases:
                candidate_fn(*case.args, **case.kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return elapsed / (config.benchmark_loops * max(len(task.benchmark_cases), 1))
