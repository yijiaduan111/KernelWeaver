"""Cheap framework-level static guards for generated candidates.

These checks catch protocol mistakes before launching KernelBench evaluation.
Backend-specific interface checks live under `src.core.contracts` so CUDA,
Triton, TileLang, and Cute can evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..utils import extract_anchor_names
from .contracts import check_backend_contract
from .hygiene import validate_python_source
from .regions import validate_region_scaffold


@dataclass
class StaticCheckResult:
    ok: bool
    failure_type: str | None = None
    logs: list[str] = field(default_factory=list)


def check_candidate_static(source_code: str, backend: str | None = None) -> StaticCheckResult:
    checks = [_check_basic_source(source_code), _check_backend_contract(source_code, backend)]
    logs: list[str] = []
    for result in checks:
        logs.extend(result.logs)
        if not result.ok:
            return StaticCheckResult(False, result.failure_type, logs)
    return StaticCheckResult(True, logs=logs)


def _check_basic_source(source_code: str) -> StaticCheckResult:
    stripped = source_code.lstrip()
    if (
        stripped.startswith("{")
        or "\"anchor_patches\"" in source_code
        or "'anchor_patches'" in source_code
        or "\"region_patches\"" in source_code
        or "'region_patches'" in source_code
    ):
        return _fail("unapplied_patch_payload", "candidate still contains an unapplied patch payload")
    if "class ModelNew" not in source_code:
        return _fail("missing_modelnew", "candidate is missing class ModelNew")
    if "def forward" not in source_code:
        return _fail("missing_forward", "candidate is missing forward method")
    syntax = validate_python_source(source_code)
    if not syntax.ok:
        return _fail(syntax.failure_type or "python_syntax_error", syntax.message or "candidate has invalid Python syntax")
    ok, message = validate_region_scaffold(source_code)
    if not ok:
        return _fail("invalid_region_scaffold", message or "candidate has invalid editable region markers")
    if not extract_anchor_names(source_code):
        return _fail("no_anchor_markers", "candidate has no anchor markers")
    return StaticCheckResult(True)


def _check_backend_contract(source_code: str, backend: str | None) -> StaticCheckResult:
    result = check_backend_contract(source_code, backend)
    return StaticCheckResult(result.ok, result.failure_type, list(result.logs))


def _fail(failure_type: str, message: str) -> StaticCheckResult:
    return StaticCheckResult(False, failure_type, [message])
