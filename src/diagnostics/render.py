from __future__ import annotations

from typing import Any

from .schema import MachineCheckProfile, TaskDiagnostics


def machine_check_profile_to_prompt_dict(profile: MachineCheckProfile | None) -> dict[str, Any] | None:
    if profile is None or not profile.enabled:
        return None
    return {
        "tier": profile.tier,
        "bottleneck_id": profile.bottleneck_id,
        "case_id": profile.case_id,
        "expected_speedup": profile.expected_speedup,
        "kernel_structure": profile.kernel_structure,
        "allowed_methods": list(profile.allowed_methods[:8]),
        "forbidden_methods": list(profile.forbidden_methods[:8]),
        "matched_predicates": list(profile.matched_predicates[:8]),
        "key_metrics": dict(profile.key_metrics),
        "code_features_used": dict(profile.code_features_used),
        "notes": list(profile.notes[:6]),
    }


def task_diagnostics_to_prompt_dict(profile: TaskDiagnostics | None) -> dict[str, Any] | None:
    if profile is None or not profile.enabled:
        return None
    ncu = profile.ncu_profile
    return {
        "mode": profile.mode,
        "ncu_profile": {
            "status": getattr(ncu, "status", "disabled"),
            "kernel_name": getattr(ncu, "kernel_name", None),
            "row_count": getattr(ncu, "row_count", 0),
            "kernel_launch_count": getattr(ncu, "kernel_launch_count", 0),
            "notes": list(getattr(ncu, "notes", [])[:4]),
        }
        if ncu is not None
        else None,
        "machine_check_profile": machine_check_profile_to_prompt_dict(profile.machine_check_profile),
        "notes": list(profile.notes[:6]),
    }
