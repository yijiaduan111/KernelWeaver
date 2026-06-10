from __future__ import annotations

from pathlib import Path
from typing import Any

from ..km_machine_check import run_machine_check
from .schema import MachineCheckProfile


DEFAULT_RULEBOOK_PATH = Path(__file__).resolve().parents[1] / "km_bottleneck.yaml"


def build_machine_check_profile(
    metric_csv_path: Path,
    *,
    kernel_name: str | None,
    code_features: dict[str, Any],
    aggregate: bool = True,
) -> MachineCheckProfile:
    try:
        payload = run_machine_check(
            DEFAULT_RULEBOOK_PATH,
            metric_csv_path,
            kernel_filter=kernel_name,
            cuda_code="",
            feature_mode="manual",
            code_features=code_features,
            aggregate=aggregate,
        )
    except Exception as exc:
        return MachineCheckProfile(
            enabled=False,
            status="error",
            source_rulebook=DEFAULT_RULEBOOK_PATH.name,
            error=str(exc),
            notes=["machine_check execution failed."],
        )

    allowed = [str(item) for item in payload.get("allowed_methods", []) if str(item).strip()]
    forbidden = [str(item) for item in payload.get("forbidden_methods", []) if str(item).strip()]
    case_id = str(payload.get("case_id", "") or "")
    status = "ok" if allowed else "no_match"
    notes: list[str] = []
    if case_id == "NO_MATCH":
        notes.append("machine_check returned no matching decision-table case.")
    return MachineCheckProfile(
        enabled=bool(allowed),
        status=status,
        source_rulebook=DEFAULT_RULEBOOK_PATH.name,
        feature_mode=str(payload.get("feature_mode", "manual")),
        tier=_optional_str(payload.get("tier")),
        bottleneck_id=_optional_str(payload.get("bottleneck_id")),
        case_id=_optional_str(payload.get("case_id")),
        expected_speedup=_optional_str(payload.get("expected_speedup")),
        kernel_structure=_optional_str(payload.get("kernel_structure")),
        kernel_structure_id=_optional_int(payload.get("kernel_structure_id")),
        allowed_methods=allowed,
        forbidden_methods=forbidden,
        matched_predicates=[str(item) for item in payload.get("matched_predicates", []) if str(item).strip()],
        code_features_used=dict(payload.get("code_features_used") or {}),
        key_metrics=dict(payload.get("key_metrics") or {}),
        missing_fields=[str(item) for item in payload.get("missing_fields", []) if str(item).strip()],
        notes=notes,
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
