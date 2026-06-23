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
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class TaskDiagnostics:
    enabled: bool = False
    mode: str = "disabled"
    ncu_profile: NcuProfile | None = None
    notes: list[str] = field(default_factory=list)


def task_diagnostics_to_dict(profile: TaskDiagnostics | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return asdict(profile)


def task_diagnostics_from_dict(payload: dict[str, Any] | None) -> TaskDiagnostics | None:
    if not payload:
        return None
    ncu_payload = payload.get("ncu_profile") if isinstance(payload.get("ncu_profile"), dict) else None
    return TaskDiagnostics(
        enabled=bool(payload.get("enabled", False)),
        mode=str(payload.get("mode", "disabled")),
        ncu_profile=_ncu_profile_from_dict(ncu_payload),
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
        raw_metrics=dict(payload.get("raw_metrics") or {}),
        notes=[str(item) for item in payload.get("notes", []) if str(item).strip()],
        error=_optional_str(payload.get("error")),
    )


def _optional_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
