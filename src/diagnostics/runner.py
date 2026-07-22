from __future__ import annotations

from .ncu import cleanup_profile_artifact, profile_candidate_with_ncu
from .schema import NcuProfile, TaskDiagnostics


def build_task_diagnostics(task, config, *, candidate_code: str | None = None, root_evaluation=None) -> TaskDiagnostics | None:
    wants_runtime_ncu = bool(
        getattr(config, "diagnostics_enabled", False) or getattr(config, "deliberation_enabled", False)
    )
    if not wants_runtime_ncu:
        return TaskDiagnostics(enabled=False, mode="disabled", notes=["Diagnostics are disabled by configuration."])
    if str(getattr(task, "benchmark_family", "") or "") != "kernelbench":
        return TaskDiagnostics(enabled=False, mode=config.diagnostics_mode, notes=["Diagnostics are only enabled for KernelBench tasks."])
    if str(getattr(task, "backend", "") or "") != "cuda":
        return TaskDiagnostics(enabled=False, mode=config.diagnostics_mode, notes=["Diagnostics are currently scoped to native CUDA tasks only."])

    if root_evaluation is not None and (
        not bool(getattr(root_evaluation, "compile_ok", False)) or not bool(getattr(root_evaluation, "correct", False))
    ):
        notes = [
            "Skipped root-candidate NCU because the root evaluation is not runnable.",
            f"compile_ok={bool(getattr(root_evaluation, 'compile_ok', False))}",
            f"correct={bool(getattr(root_evaluation, 'correct', False))}",
            f"failure_stage={getattr(root_evaluation, 'failure_stage', 'none')}",
        ]
        return TaskDiagnostics(
            enabled=False,
            mode=config.diagnostics_mode,
            ncu_profile=NcuProfile(enabled=False, status="skipped", notes=list(notes)),
            notes=list(notes),
        )

    candidate_source = candidate_code if candidate_code is not None else getattr(task, "source_code", None)
    csv_path = None
    try:
        ncu_profile, csv_path = profile_candidate_with_ncu(
            task,
            str(candidate_source or ""),
            timeout_seconds=int(getattr(config, "diagnostics_timeout_seconds", 300)),
            warmup_runs=int(getattr(config, "diagnostics_warmup_runs", 2)),
            profile_runs=int(getattr(config, "diagnostics_profile_runs", 3)),
        )
        return TaskDiagnostics(
            enabled=bool(ncu_profile.enabled),
            mode=config.diagnostics_mode,
            ncu_profile=ncu_profile,
            notes=list(ncu_profile.notes),
        )
    except Exception as exc:
        notes = [
            "NCU diagnostics failed before search and were downgraded to unavailable.",
            "Continuing without NCU diagnostics for this task.",
        ]
        ncu_profile = NcuProfile(
            enabled=False,
            status="error",
            notes=notes,
            error=f"{type(exc).__name__}: {str(exc)[:1000]}",
        )
        return TaskDiagnostics(
            enabled=False,
            mode=config.diagnostics_mode,
            ncu_profile=ncu_profile,
            notes=notes,
        )
    finally:
        cleanup_profile_artifact(csv_path)
