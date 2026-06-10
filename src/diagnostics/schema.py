from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NcuProfile:
    enabled: bool = False
    status: str = "disabled"
    profiler: str = "ncu"
    kernel_name: str | None = None
    row_count: int = 0
    kernel_launch_count: int = 0
    notes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class MachineCheckProfile:
    enabled: bool = False
    status: str = "disabled"
    source_rulebook: str = "km_bottleneck.yaml"
    feature_mode: str = "manual"
    tier: str | None = None
    bottleneck_id: str | None = None
    case_id: str | None = None
    expected_speedup: str | None = None
    kernel_structure: str | None = None
    kernel_structure_id: int | None = None
    allowed_methods: list[str] = field(default_factory=list)
    forbidden_methods: list[str] = field(default_factory=list)
    matched_predicates: list[str] = field(default_factory=list)
    code_features_used: dict[str, Any] = field(default_factory=dict)
    key_metrics: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class TaskDiagnostics:
    enabled: bool = False
    mode: str = "disabled"
    ncu_profile: NcuProfile | None = None
    machine_check_profile: MachineCheckProfile | None = None
    notes: list[str] = field(default_factory=list)


def machine_check_profile_to_dict(profile: MachineCheckProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return asdict(profile)


def machine_check_profile_from_dict(payload: dict[str, Any] | None) -> MachineCheckProfile | None:
    if not payload:
        return None
    return MachineCheckProfile(
        enabled=bool(payload.get("enabled", False)),
        status=str(payload.get("status", "disabled")),
        source_rulebook=str(payload.get("source_rulebook", "km_bottleneck.yaml")),
        feature_mode=str(payload.get("feature_mode", "manual")),
        tier=_optional_str(payload.get("tier")),
        bottleneck_id=_optional_str(payload.get("bottleneck_id")),
        case_id=_optional_str(payload.get("case_id")),
        expected_speedup=_optional_str(payload.get("expected_speedup")),
        kernel_structure=_optional_str(payload.get("kernel_structure")),
        kernel_structure_id=_optional_int(payload.get("kernel_structure_id")),
        allowed_methods=[str(item) for item in payload.get("allowed_methods", []) if str(item).strip()],
        forbidden_methods=[str(item) for item in payload.get("forbidden_methods", []) if str(item).strip()],
        matched_predicates=[str(item) for item in payload.get("matched_predicates", []) if str(item).strip()],
        code_features_used=dict(payload.get("code_features_used") or {}),
        key_metrics=dict(payload.get("key_metrics") or {}),
        missing_fields=[str(item) for item in payload.get("missing_fields", []) if str(item).strip()],
        notes=[str(item) for item in payload.get("notes", []) if str(item).strip()],
        error=_optional_str(payload.get("error")),
    )


def task_diagnostics_to_dict(profile: TaskDiagnostics | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    payload = asdict(profile)
    return payload


def task_diagnostics_from_dict(payload: dict[str, Any] | None) -> TaskDiagnostics | None:
    if not payload:
        return None
    ncu_payload = payload.get("ncu_profile") if isinstance(payload.get("ncu_profile"), dict) else None
    machine_payload = payload.get("machine_check_profile") if isinstance(payload.get("machine_check_profile"), dict) else None
    return TaskDiagnostics(
        enabled=bool(payload.get("enabled", False)),
        mode=str(payload.get("mode", "disabled")),
        ncu_profile=_ncu_profile_from_dict(ncu_payload),
        machine_check_profile=machine_check_profile_from_dict(machine_payload),
        notes=[str(item) for item in payload.get("notes", []) if str(item).strip()],
    )


def _ncu_profile_from_dict(payload: dict[str, Any] | None) -> NcuProfile | None:
    if not payload:
        return None
    return NcuProfile(
        enabled=bool(payload.get("enabled", False)),
        status=str(payload.get("status", "disabled")),
        profiler=str(payload.get("profiler", "ncu")),
        kernel_name=_optional_str(payload.get("kernel_name")),
        row_count=int(payload.get("row_count", 0) or 0),
        kernel_launch_count=int(payload.get("kernel_launch_count", 0) or 0),
        notes=[str(item) for item in payload.get("notes", []) if str(item).strip()],
        error=_optional_str(payload.get("error")),
    )


def _optional_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
