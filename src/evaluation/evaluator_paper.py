from __future__ import annotations

"""KernelBench evaluator aligned with the official open-source helpers."""

import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Callable

from ..backends import is_supported_kernelbench_backend, normalize_backend, supported_kernelbench_backends_text
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
        # A bad generated kernel can leave CUDA's context in a sticky illegal-access
        # state.  Evaluate every root/candidate in a fresh spawned process so one
        # candidate cannot invalidate the remaining search or later benchmark tasks.
        if os.environ.get("STARK_PAPER_EVAL_ISOLATION", "1").strip().lower() not in {"0", "false", "no"}:
            return _evaluate_isolated(task, code, config)
        return self._evaluate_in_process(task, code, config)

    def _evaluate_in_process(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        if task.entry_kind != "model_class":
            return _failure_result(
                "compile",
                "compile_error",
                f"KernelBenchPaperEvaluator only supports model_class tasks, got entry_kind={task.entry_kind}",
            )
        backend = normalize_backend(task.backend or "triton")
        if not is_supported_kernelbench_backend(backend):
            return _failure_result(
                "compile",
                "compile_error",
                f"KernelBenchPaperEvaluator only supports {supported_kernelbench_backends_text()} backends, got backend={backend}",
            )

        torch = _require_torch()
        if not torch.cuda.is_available():
            return _failure_result("runtime", "runtime_error", "runtime_error[cuda]: CUDA is not available")

        try:
            eval_kernel_against_ref, measure_ref_program_time = _load_official_kernelbench_symbols(task)
        except Exception as exc:
            return _failure_result("compile", "paper_evaluator_import_error", f"paper_evaluator_import_error: {exc}")

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        eval_precision, ref_precision = _precision_settings(torch, backend)
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
                precision=eval_precision,
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
            ref_precision=ref_precision,
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
        ref_precision: str,
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
                    ref_precision=ref_precision,
                )
            elif mode == "torch_compile_default":
                reference_runtimes[mode] = self._measure_reference_mode(
                    task,
                    config,
                    measure_ref_program_time,
                    device=device,
                    use_torch_compile=True,
                    compile_mode="default",
                    ref_precision=ref_precision,
                )
            elif mode == "torch_compile_max_autotune":
                reference_runtimes[mode] = self._measure_reference_mode(
                    task,
                    config,
                    measure_ref_program_time,
                    device=device,
                    use_torch_compile=True,
                    compile_mode="max-autotune",
                    ref_precision=ref_precision,
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
        ref_precision: str,
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
                precision=ref_precision,
            )
        except Exception:
            return None
        if not isinstance(stats, dict):
            return None
        return _seconds_from_milliseconds(stats.get("mean"))


def _precision_settings(torch, backend: str) -> tuple[Any, str]:
    if backend in {"tilelang", "cute"}:
        return torch.bfloat16, "bf16"
    return torch.float32, "fp32"


def _paper_exception_stage(torch, exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()
    error_name = exc.__class__.__name__.lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "runtime", "cuda_out_of_memory"
    if _looks_like_compile_failure(message):
        return "compile", "cuda_compile_error"
    runtime_markers = (
        "cuda error:",
        "cuda kernel launch",
        "device-side",
        "illegal memory access",
        "misaligned address",
        "invalid configuration argument",
    )
    if any(marker in message for marker in runtime_markers):
        return "runtime", error_name or "runtime_error"
    if isinstance(exc, (SyntaxError, ImportError, ModuleNotFoundError)):
        return "compile", error_name or "compile_error"
    return "compile", error_name or "paper_eval_error"


def _paper_failure_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
    # KernelBench imports ModelNew before its load_inline extension is built.
    # A lazy nvcc/ninja failure therefore arrives under runtime_error metadata
    # even though the candidate never produced an executable CUDA kernel.
    diagnostic = "\n".join(
        str(metadata.get(key) or "")
        for key in (
            "compilation_error_name",
            "compilation_error",
            "runtime_error_name",
            "runtime_error",
            "runtime_error_traceback",
        )
    ).lower()
    if _looks_like_compile_failure(diagnostic):
        return "compile", "cuda_compile_error"
    if metadata.get("runtime_error") is not None or metadata.get("runtime_error_name") is not None:
        return "runtime", str(metadata.get("runtime_error_name") or "runtime_error")
    if metadata.get("correctness_issue") is not None or metadata.get("correctness_issue_name") is not None:
        return "correctness", str(metadata.get("correctness_issue_name") or "correctness_error")
    if metadata.get("compilation_error") is not None or metadata.get("compilation_error_name") is not None:
        return "compile", str(metadata.get("compilation_error_name") or "compile_error")
    return "correctness", "correctness_error"


def _looks_like_compile_failure(message: str) -> bool:
    compile_markers = (
        "error building extension",
        "ninja: build stopped",
        "nvcc fatal",
        "ptxas error",
        "compilation error",
        "failed: cuda.cuda.o",
    )
    return any(marker in message for marker in compile_markers)


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


def _paper_eval_worker(connection, task: TaskSpec, code: str, config: StarkConfig) -> None:
    try:
        result = KernelBenchPaperEvaluator()._evaluate_in_process(task, code, config)
        connection.send(("ok", result))
    except BaseException as exc:
        connection.send(("error", f"{exc.__class__.__name__}: {exc}"))
    finally:
        connection.close()


def _paper_eval_timeout_seconds(config: StarkConfig) -> int:
    configured = int(getattr(config, "paper_eval_timeout_seconds", 900))
    override = os.environ.get("STARK_PAPER_EVAL_TIMEOUT_SECONDS")
    return max(60, int(override) if override is not None else configured)


def _evaluate_isolated(task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_paper_eval_worker,
        args=(send_connection, task, code, config),
        daemon=False,
    )
    process.start()
    send_connection.close()
    timeout_seconds = _paper_eval_timeout_seconds(config)
    try:
        if receive_connection.poll(timeout_seconds):
            status, payload = receive_connection.recv()
            process.join(timeout=10)
            if status == "ok" and isinstance(payload, EvaluationResult):
                return payload
            return _failure_result("runtime", "isolated_evaluator_error", f"isolated_evaluator_error: {payload}")
        process.terminate()
        process.join(timeout=10)
        return _failure_result(
            "runtime",
            "isolated_evaluator_timeout",
            f"isolated_evaluator_timeout: exceeded {timeout_seconds}s",
        )
    except (EOFError, OSError) as exc:
        process.join(timeout=10)
        return _failure_result(
            "runtime",
            "isolated_evaluator_crash",
            f"isolated_evaluator_crash: exitcode={process.exitcode}; {exc}",
        )
    finally:
        receive_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
