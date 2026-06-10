from __future__ import annotations

from .code_features import derive_static_code_features
from .machine_check import build_machine_check_profile
from .ncu import cleanup_profile_artifact, profile_reference_with_ncu
from .schema import MachineCheckProfile, TaskDiagnostics


def build_task_diagnostics(task, config) -> TaskDiagnostics | None:
    if not getattr(config, "diagnostics_enabled", False):
        return TaskDiagnostics(enabled=False, mode="disabled", notes=["Diagnostics are disabled by configuration."])
    if str(getattr(task, "benchmark_family", "") or "") != "kernelbench":
        return TaskDiagnostics(enabled=False, mode=config.diagnostics_mode, notes=["Diagnostics are only enabled for KernelBench tasks."])
    if str(getattr(task, "backend", "") or "") != "cuda":
        return TaskDiagnostics(enabled=False, mode=config.diagnostics_mode, notes=["Diagnostics are currently scoped to native CUDA tasks only."])

    feature_result = derive_static_code_features(task)
    if not feature_result.supported:
        return TaskDiagnostics(
            enabled=False,
            mode=config.diagnostics_mode,
            notes=list(feature_result.notes),
            machine_check_profile=MachineCheckProfile(
                enabled=False,
                status="unsupported",
                notes=list(feature_result.notes),
            ),
        )

    ncu_profile, csv_path = profile_reference_with_ncu(
        task,
        timeout_seconds=int(getattr(config, "diagnostics_timeout_seconds", 300)),
        warmup_runs=int(getattr(config, "diagnostics_warmup_runs", 2)),
        profile_runs=int(getattr(config, "diagnostics_profile_runs", 3)),
    )
    try:
        if csv_path is None:
            return TaskDiagnostics(
                enabled=False,
                mode=config.diagnostics_mode,
                ncu_profile=ncu_profile,
                notes=list(feature_result.notes) + list(ncu_profile.notes),
            )
        machine_check = build_machine_check_profile(
            csv_path,
            kernel_name=ncu_profile.kernel_name,
            code_features=feature_result.features,
            aggregate=True,
        )
        enabled = bool(ncu_profile.enabled and machine_check.enabled)
        notes = list(feature_result.notes) + list(ncu_profile.notes) + list(machine_check.notes)
        return TaskDiagnostics(
            enabled=enabled,
            mode=config.diagnostics_mode,
            ncu_profile=ncu_profile,
            machine_check_profile=machine_check,
            notes=notes,
        )
    finally:
        cleanup_profile_artifact(csv_path)
