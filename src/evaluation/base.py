from __future__ import annotations

import importlib.util
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..models import EvaluationResult, TaskSpec, TestCase
from ..utils import clone_value


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, task: TaskSpec, code: str, config) -> EvaluationResult:
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


def _build_speedups(candidate_runtime: float | None, reference_runtimes: dict[str, float | None]) -> dict[str, float | None]:
    speedups: dict[str, float | None] = {}
    for name, reference_runtime in reference_runtimes.items():
        if candidate_runtime is None or reference_runtime is None or candidate_runtime <= 0 or reference_runtime <= 0:
            speedups[name] = None
        else:
            speedups[name] = reference_runtime / candidate_runtime
    return speedups


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
