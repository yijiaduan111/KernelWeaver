from __future__ import annotations

"""KernelBench evaluator aligned with the official open-source helpers."""

import sys
from pathlib import Path
from typing import Any, Callable

from ..models import EvaluationResult, StarkConfig, TaskSpec
from .base import PRIMARY_REFERENCE, REFERENCE_MODES, Evaluator, _build_speedups, _failure_result, _require_torch, _success_result


def _load_official_kernelbench_symbols(task: TaskSpec | None = None) -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from kernelbench.eval import eval_kernel_against_ref
        from kernelbench.timing import measure_ref_program_time
        return eval_kernel_against_ref, measure_ref_program_time
    except ModuleNotFoundError:
        _ensure_kernelbench_on_sys_path(task)
        from kernelbench.eval import eval_kernel_against_ref
        from kernelbench.timing import measure_ref_program_time
        return eval_kernel_against_ref, measure_ref_program_time


def _ensure_kernelbench_on_sys_path(task: TaskSpec | None = None) -> None:
    candidates: list[Path] = []
    if task is not None and getattr(task, "source_root", None):
        source_root = Path(str(task.source_root))
        candidates.extend([source_root / "src", source_root])
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _selected_reference_modes(config: StarkConfig) -> list[str]:
    raw_modes = list(getattr(config, "reference_modes", []) or [PRIMARY_REFERENCE])
    filtered = [mode for mode in REFERENCE_MODES if mode in raw_modes]
    return filtered or [PRIMARY_REFERENCE]


class KernelBenchPaperEvaluator(Evaluator):
    """Evaluate one candidate with the official KernelBench correctness and timing path."""

    def evaluate(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        if task.entry_kind != "model_class":
            return _failure_result(
                "compile",
                "compile_error",
                f"KernelBenchPaperEvaluator only supports model_class tasks, got entry_kind={task.entry_kind}",
            )
        backend = str(task.backend or "triton").lower()
        if backend not in {"triton", "cuda"}:
            return _failure_result(
                "compile",
                "compile_error",
                f"KernelBenchPaperEvaluator only supports triton/cuda backends, got backend={backend}",
            )

        torch = _require_torch()
        if not torch.cuda.is_available():
            return _failure_result("runtime", "runtime_error", "runtime_error[cuda]: CUDA is not available")

        try:
            eval_kernel_against_ref, measure_ref_program_time = _load_official_kernelbench_symbols(task)
        except Exception as exc:
            return _failure_result("compile", "paper_evaluator_import_error", f"paper_evaluator_import_error: {exc}")

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        try:
            official_result = eval_kernel_against_ref(
                original_model_src=task.reference_code,
                custom_model_src=code,
                seed_num=config.seed,
                num_correct_trials=config.num_correct_trials,
                num_perf_trials=config.num_perf_trials,
                measure_performance=True,
                timing_method=config.timing_method,
                verbose=config.verbose,
                device=device,
                backend=backend,
                precision=torch.float32,
                check_for_excessive_speedup=False,
            )
        except Exception as exc:
            failure_stage, failure_type = _paper_exception_stage(torch, exc)
            return _failure_result(failure_stage, failure_type, f"paper_eval_error: {exc}")

        if official_result is None:
            return _failure_result(
                "compile",
                "paper_eval_retryable_compile_error",
                "paper_eval returned None; official KernelBench treats this as a retryable compile/load failure.",
            )

        metadata = dict(getattr(official_result, "metadata", {}) or {})
        if not bool(getattr(official_result, "compiled", False)):
            failure_type = str(metadata.get("compilation_error_name") or "compile_error")
            return _failure_result("compile", failure_type, *_metadata_logs(metadata))

        if not bool(getattr(official_result, "correctness", False)):
            failure_stage, failure_type = _paper_failure_from_metadata(metadata)
            return _failure_result(failure_stage, failure_type, *_metadata_logs(metadata))

        candidate_runtime = _seconds_from_milliseconds(getattr(official_result, "runtime", None))
        if candidate_runtime is None:
            logs = _metadata_logs(metadata)
            logs.append("paper_eval_missing_runtime=true")
            return _failure_result("runtime", "runtime_error", *logs)

        eager_fallback = _seconds_from_milliseconds(getattr(official_result, "ref_runtime", None))
        reference_runtimes = self._measure_reference_runtimes(
            task=task,
            config=config,
            measure_ref_program_time=measure_ref_program_time,
            device=device,
            eager_fallback=eager_fallback,
        )
        speedups = _build_speedups(candidate_runtime, reference_runtimes)
        primary_reference = PRIMARY_REFERENCE if PRIMARY_REFERENCE in reference_runtimes else next(iter(reference_runtimes.keys()), PRIMARY_REFERENCE)
        result = _success_result(
            candidate_runtime,
            reference_runtime=reference_runtimes.get(primary_reference),
            speedup=speedups.get(primary_reference),
            reference_runtimes=reference_runtimes,
            speedups=speedups,
            primary_reference=primary_reference,
        )
        result.logs.extend(_metadata_logs(metadata))
        return result

    def _measure_reference_runtimes(
        self,
        task: TaskSpec,
        config: StarkConfig,
        measure_ref_program_time: Callable[..., Any],
        device,
        eager_fallback: float | None,
    ) -> dict[str, float | None]:
        reference_runtimes: dict[str, float | None] = {}
        for mode in _selected_reference_modes(config):
            if mode == PRIMARY_REFERENCE:
                reference_runtimes[mode] = self._measure_reference_mode(
                    task,
                    config,
                    measure_ref_program_time,
                    device=device,
                    use_torch_compile=False,
                    compile_mode=None,
                )
            elif mode == "torch_compile_default":
                reference_runtimes[mode] = self._measure_reference_mode(
                    task,
                    config,
                    measure_ref_program_time,
                    device=device,
                    use_torch_compile=True,
                    compile_mode="default",
                )
            elif mode == "torch_compile_max_autotune":
                reference_runtimes[mode] = self._measure_reference_mode(
                    task,
                    config,
                    measure_ref_program_time,
                    device=device,
                    use_torch_compile=True,
                    compile_mode="max-autotune",
                )
        if PRIMARY_REFERENCE in reference_runtimes and reference_runtimes[PRIMARY_REFERENCE] is None:
            reference_runtimes[PRIMARY_REFERENCE] = eager_fallback
        return reference_runtimes

    @staticmethod
    def _measure_reference_mode(
        task: TaskSpec,
        config: StarkConfig,
        measure_ref_program_time: Callable[..., Any],
        device,
        use_torch_compile: bool,
        compile_mode: str | None,
    ) -> float | None:
        try:
            stats = measure_ref_program_time(
                ref_arch_name=task.name,
                ref_arch_src=task.reference_code,
                num_warmup=config.paper_num_warmup,
                num_trials=config.num_perf_trials,
                discard_first=config.paper_discard_first,
                timing_method=config.timing_method,
                use_torch_compile=use_torch_compile,
                torch_compile_options=compile_mode or "default",
                device=device,
                verbose=config.verbose,
                precision="fp32",
            )
        except Exception:
            return None
        if not isinstance(stats, dict):
            return None
        return _seconds_from_milliseconds(stats.get("mean"))


def _paper_exception_stage(torch, exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()
    error_name = exc.__class__.__name__.lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "runtime", "cuda_out_of_memory"
    if "cuda" in message or "device-side" in message or "illegal memory access" in message:
        return "runtime", error_name or "runtime_error"
    if isinstance(exc, (SyntaxError, ImportError, ModuleNotFoundError)):
        return "compile", error_name or "compile_error"
    return "compile", error_name or "paper_eval_error"


def _paper_failure_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
    if metadata.get("runtime_error") is not None or metadata.get("runtime_error_name") is not None:
        return "runtime", str(metadata.get("runtime_error_name") or "runtime_error")
    if metadata.get("correctness_issue") is not None or metadata.get("correctness_issue_name") is not None:
        return "correctness", str(metadata.get("correctness_issue_name") or "correctness_error")
    if metadata.get("compilation_error") is not None or metadata.get("compilation_error_name") is not None:
        return "compile", str(metadata.get("compilation_error_name") or "compile_error")
    return "correctness", "correctness_error"


def _metadata_logs(metadata: dict[str, Any]) -> list[str]:
    ordered_keys = [
        "compilation_error_name",
        "compilation_error",
        "runtime_error_name",
        "runtime_error",
        "correctness_issue_name",
        "correctness_issue",
        "max_difference",
        "avg_difference",
        "excessive_speedup",
        "error_during_performance",
    ]
    logs: list[str] = []
    for key in ordered_keys:
        if key in metadata and metadata[key] is not None:
            logs.append(f"{key}={metadata[key]}")
    traceback_value = metadata.get("runtime_error_traceback")
    if traceback_value:
        last_line = str(traceback_value).strip().splitlines()[-1]
        logs.append(f"runtime_error_traceback={last_line}")
    return logs


def _seconds_from_milliseconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value / 1000.0
