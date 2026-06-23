from __future__ import annotations

from typing import Any

from .schema import NcuProfile, TaskDiagnostics


def ncu_profile_to_prompt_dict(profile: NcuProfile | None) -> dict[str, Any] | None:
    if profile is None or not profile.enabled:
        return None
    return {
        "status": profile.status,
        "kernel_name": profile.kernel_name,
        "row_count": profile.row_count,
        "kernel_launch_count": profile.kernel_launch_count,
        "raw_metrics": dict(profile.raw_metrics),
        "notes": list(profile.notes[:4]),
    }


def task_diagnostics_to_prompt_dict(profile: TaskDiagnostics | None) -> dict[str, Any] | None:
    if profile is None or not profile.enabled:
        return None
    return {
        "mode": profile.mode,
        "ncu_profile": ncu_profile_to_prompt_dict(profile.ncu_profile),
        "notes": list(profile.notes[:6]),
    }
