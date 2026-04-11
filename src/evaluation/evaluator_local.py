from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..models import EvaluationResult, StarkConfig, TaskSpec, TestCase
from ..utils import clone_value, compare_values


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        raise NotImplementedError


PRIMARY_REFERENCE = "torch_eager"
REFERENCE_MODES = (
    PRIMARY_REFERENCE,
    "torch_compile_default",
    "torch_compile_max_autotune",
)


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
        primary_reference=PRIMARY_REFERENCE,
    )


def _success_result(
    runtime: float,
    reference_runtime: float | None = None,
    speedup: float | None = None,
    reference_runtimes: dict[str, float | None] | None = None,
    speedups: dict[str, float | None] | None = None,
    primary_reference: str | None = PRIMARY_REFERENCE,
) -> EvaluationResult:
    reference_runtime_map = dict(reference_runtimes or {})
    speedup_map = dict(speedups or {})
    if primary_reference is not None:
        if primary_reference not in reference_runtime_map:
            reference_runtime_map[primary_reference] = reference_runtime
        if primary_reference not in speedup_map:
            speedup_map[primary_reference] = speedup
        if reference_runtime is None:
            reference_runtime = reference_runtime_map.get(primary_reference)
        if speedup is None:
            speedup = speedup_map.get(primary_reference)
    return EvaluationResult(
        compile_ok=True,
        correct=True,
        runtime=runtime,
        score=runtime,
        logs=[],
        failure_type=None,
        failure_stage="none",
        reference_runtime=reference_runtime,
        speedup=speedup,
        reference_runtimes=reference_runtime_map,
        speedups=speedup_map,
        primary_reference=primary_reference,
    )


def _require_local_bridge_cases(task: TaskSpec, evaluator_name: str) -> EvaluationResult | None:
    if task.test_cases and task.benchmark_cases:
        return None
    return _failure_result(
        "compile",
        "bridge_cases_missing",
        (
            f"{evaluator_name} requires local test and benchmark cases. "
            "Auto-bridged KernelBench tasks should use the paper evaluator, "
            "or add a curated bridge override with local cases."
        ),
    )


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


class KernelBenchEvaluator(Evaluator):
    def evaluate(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        if task.entry_kind != "model_class":
            return _failure_result("compile", "compile_error", f"KernelBenchEvaluator only supports model_class tasks, got entry_kind={task.entry_kind}")
        bridge_error = _require_local_bridge_cases(task, "KernelBenchEvaluator")
        if bridge_error is not None:
            return bridge_error
        torch = _require_torch()
        if not torch.cuda.is_available():
            return _failure_result("runtime", "runtime_error", "runtime_error[cuda]: CUDA is not available")
        try:
            with _load_module_from_source(code, prefix=f"candidate_{task.name}") as candidate_module:
                with _load_module_from_source(task.reference_code, prefix=f"reference_{task.name}") as reference_module:
                    candidate_cls = getattr(candidate_module, task.function_name, None)
                    reference_cls = getattr(reference_module, task.reference_function_name, None)
                    if candidate_cls is None or not callable(candidate_cls):
                        raise ValueError(f"Entrypoint '{task.function_name}' is missing.")
                    if reference_cls is None or not callable(reference_cls):
                        raise ValueError(f"Reference entrypoint '{task.reference_function_name}' is missing.")
                    for case in task.test_cases:
                        try:
                            actual = self._run_model(candidate_cls, case, config.seed)
                            expected = self._run_model(reference_cls, case, config.seed)
                        except Exception as exc:
                            return _failure_result("runtime", "runtime_error", f"runtime_error[{case.label}]: {exc}")
                        if not compare_values(actual, expected, tolerance=1e-4):
                            return _failure_result("correctness", "correctness_error", f"correctness_error[{case.label}]")
                    try:
                        reference_runtimes = self._measure_reference_runtimes(reference_cls, task.benchmark_cases, config, config.seed)
                        candidate_runtime = self._measure_model_runtime(candidate_cls, task.benchmark_cases, config, config.seed)
                    except Exception as exc:
                        return _failure_result("runtime", "runtime_error", f"runtime_error[benchmark]: {exc}")
        except Exception as exc:
            return _failure_result("compile", "compile_error", f"compile_error: {exc}")
        speedups = _build_speedups(candidate_runtime, reference_runtimes)
        return _success_result(
            candidate_runtime,
            reference_runtime=reference_runtimes.get(PRIMARY_REFERENCE),
            speedup=speedups.get(PRIMARY_REFERENCE),
            reference_runtimes=reference_runtimes,
            speedups=speedups,
            primary_reference=PRIMARY_REFERENCE,
        )

    @staticmethod
    def _run_model(model_cls, case: TestCase, seed: int):
        torch = _require_torch()
        model = _instantiate_model(model_cls, case, seed, device="cuda")
        args = _move_to_device(_clone_args(case.args), device="cuda")
        kwargs = _move_to_device(_clone_kwargs(case.kwargs), device="cuda")
        with torch.no_grad():
            output = model(*args, **kwargs)
            torch.cuda.synchronize()
        return output

    @staticmethod
    def _measure_model_runtime(
        model_cls,
        cases: list[TestCase],
        config: StarkConfig,
        seed: int,
        compile_mode: str | None = None,
    ) -> float:
        torch = _require_torch()
        total_elapsed = 0.0
        measured_cases = 0
        for case in cases:
            model = _instantiate_model(model_cls, case, seed, device="cuda")
            if compile_mode is not None:
                model = _compile_model(model, compile_mode)
            args = _move_to_device(_clone_args(case.args), device="cuda")
            kwargs = _move_to_device(_clone_kwargs(case.kwargs), device="cuda")
            with torch.no_grad():
                for _ in range(config.warmup_loops):
                    model(*args, **kwargs)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(config.benchmark_loops):
                    model(*args, **kwargs)
                torch.cuda.synchronize()
            total_elapsed += time.perf_counter() - started
            measured_cases += 1
        if measured_cases == 0:
            raise RuntimeError("No benchmark cases were provided.")
        return total_elapsed / (config.benchmark_loops * measured_cases)

    @classmethod
    def _measure_reference_runtimes(
        cls,
        model_cls,
        cases: list[TestCase],
        config: StarkConfig,
        seed: int,
    ) -> dict[str, float | None]:
        runtimes = {
            PRIMARY_REFERENCE: cls._measure_model_runtime(model_cls, cases, config, seed),
            "torch_compile_default": None,
            "torch_compile_max_autotune": None,
        }
        for name, mode in (
            ("torch_compile_default", "default"),
            ("torch_compile_max_autotune", "max-autotune"),
        ):
            try:
                runtimes[name] = cls._measure_model_runtime(model_cls, cases, config, seed, compile_mode=mode)
            except Exception:
                runtimes[name] = None
        return runtimes


class CudaEvaluator(Evaluator):
    def evaluate(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        if task.entry_kind != "model_class":
            return _failure_result("compile", "compile_error", f"CudaEvaluator only supports model_class tasks, got entry_kind={task.entry_kind}")
        bridge_error = _require_local_bridge_cases(task, "CudaEvaluator")
        if bridge_error is not None:
            return bridge_error
        torch = _require_torch()
        if not torch.cuda.is_available():
            return _failure_result("runtime", "runtime_error", "runtime_error[cuda]: CUDA is not available")
        try:
            with _load_module_from_source(code, prefix=f"candidate_{task.name}") as candidate_module:
                candidate_cls = getattr(candidate_module, task.function_name, None)
                if candidate_cls is None or not callable(candidate_cls):
                    raise ValueError(f"Entrypoint '{task.function_name}' is missing.")
                self._compile_extension(candidate_module)
                with _load_module_from_source(task.reference_code, prefix=f"reference_{task.name}") as reference_module:
                    reference_cls = getattr(reference_module, task.reference_function_name, None)
                    if reference_cls is None or not callable(reference_cls):
                        raise ValueError(f"Reference entrypoint '{task.reference_function_name}' is missing.")
                    for case in task.test_cases:
                        try:
                            actual = KernelBenchEvaluator._run_model(candidate_cls, case, config.seed)
                            expected = KernelBenchEvaluator._run_model(reference_cls, case, config.seed)
                        except Exception as exc:
                            return _failure_result("runtime", "runtime_error", f"runtime_error[{case.label}]: {exc}")
                        if not compare_values(actual, expected, tolerance=1e-4):
                            return _failure_result("correctness", "correctness_error", f"correctness_error[{case.label}]")
                    try:
                        reference_runtimes = KernelBenchEvaluator._measure_reference_runtimes(
                            reference_cls,
                            task.benchmark_cases,
                            config,
                            config.seed,
                        )
                        candidate_runtime = KernelBenchEvaluator._measure_model_runtime(candidate_cls, task.benchmark_cases, config, config.seed)
                    except Exception as exc:
                        return _failure_result("runtime", "runtime_error", f"runtime_error[benchmark]: {exc}")
        except Exception as exc:
            return _failure_result("compile", "compile_error", f"compile_error: {exc}")
        speedups = _build_speedups(candidate_runtime, reference_runtimes)
        return _success_result(
            candidate_runtime,
            reference_runtime=reference_runtimes.get(PRIMARY_REFERENCE),
            speedup=speedups.get(PRIMARY_REFERENCE),
            reference_runtimes=reference_runtimes,
            speedups=speedups,
            primary_reference=PRIMARY_REFERENCE,
        )

    @staticmethod
    def _compile_extension(module: Any):
        builder = getattr(module, "_stark_get_extension", None)
        if builder is None or not callable(builder):
            raise ValueError("Candidate module is missing the required native CUDA extension builder '_stark_get_extension'.")
        return builder()


@contextmanager
def _load_module_from_source(source_code: str, prefix: str) -> Iterator[Any]:
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"{prefix}_", dir=cwd) as temp_dir:
        module_path = Path(temp_dir) / "candidate_module.py"
        module_path.write_text(source_code, encoding="utf-8")
        module_name = f"{prefix}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to build import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(module_name, None)


@contextmanager
def _load_callable_from_source(source_code: str, function_name: str, prefix: str) -> Iterator[Any]:
    with _load_module_from_source(source_code, prefix=prefix) as module:
        fn = getattr(module, function_name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Entrypoint '{function_name}' is missing.")
        yield fn


def _clone_args(args: list[Any]) -> list[Any]:
    return [clone_value(arg) for arg in args]


def _clone_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: clone_value(value) for key, value in kwargs.items()}


def _move_to_device(value: Any, device: str):
    torch = _maybe_import_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _instantiate_model(model_cls, case: TestCase, seed: int, device: str):
    torch = _require_torch()
    with _torch_seed(torch, seed):
        model = model_cls(*_clone_args(case.init_args), **_clone_kwargs(case.init_kwargs))
    model = model.to(device)
    model.eval()
    return model


def _compile_model(model, mode: str):
    torch = _require_torch()
    compiler = getattr(torch, "compile", None)
    if compiler is None or not callable(compiler):
        raise RuntimeError("torch.compile is not available in this runtime.")
    try:
        return compiler(model, mode=mode)
    except TypeError:
        return compiler(model)


def _build_speedups(candidate_runtime: float | None, reference_runtimes: dict[str, float | None]) -> dict[str, float | None]:
    speedups: dict[str, float | None] = {}
    for name, reference_runtime in reference_runtimes.items():
        if (
            candidate_runtime is None
            or reference_runtime is None
            or candidate_runtime <= 0
            or reference_runtime <= 0
        ):
            speedups[name] = None
        else:
            speedups[name] = reference_runtime / candidate_runtime
    return speedups


@contextmanager
def _torch_seed(torch_module, seed: int):
    cpu_state = torch_module.random.get_rng_state()
    cuda_state = torch_module.cuda.get_rng_state_all() if torch_module.cuda.is_available() else None
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch_module.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch_module.cuda.set_rng_state_all(cuda_state)


def _maybe_import_torch():
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"torch is required for evaluator execution: {exc}") from exc
    return torch
