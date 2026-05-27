"""Candidate normalization before evaluation.

The workflow accepts model output, not trusted source code. This module turns
supported candidate formats into a Python source file while rejecting obvious
protocol mistakes before they reach the expensive KernelBench evaluator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .regions import RegionPatch, apply_region_patches


@dataclass
class CandidateNormalizeResult:
    code: str
    ok: bool
    failure_type: str | None = None
    logs: list[str] = field(default_factory=list)


def normalize_candidate(parent_code: str, raw_output: str) -> CandidateNormalizeResult:
    """Return source code from JSON region patches or compatible full-file output."""
    text = _strip_code_fences(str(raw_output or "")).strip()
    if not text:
        return _invalid(parent_code, "empty_candidate", "candidate output is empty")

    patch_payload = _try_parse_patch_payload(text)
    if patch_payload is not None:
        return _apply_patch_payload(parent_code, patch_payload)

    if _looks_like_unapplied_patch(text):
        return _invalid(parent_code, "invalid_patch_format", "candidate looks like a patch JSON but could not be parsed as region_patches")

    if _looks_like_python_module(text):
        if _has_editable_regions(parent_code):
            return _invalid(parent_code, "full_module_region_task", "region-scaffold tasks require region_patches; full-module output is not accepted")
        return CandidateNormalizeResult(code=text, ok=True)

    return _invalid(parent_code, "invalid_candidate_format", "candidate is neither Python source nor valid region_patches JSON")


def _apply_patch_payload(parent_code: str, payload: dict[str, Any]) -> CandidateNormalizeResult:
    if "region_patches" not in payload:
        return _invalid(parent_code, "invalid_patch_format", "candidate patch payload must use region_patches")
    return _apply_region_patch_payload(parent_code, payload.get("region_patches"))


def _apply_region_patch_payload(
    parent_code: str,
    patches: Any,
    failure_type: str = "invalid_region_patch_format",
    patch_label: str = "region patch",
) -> CandidateNormalizeResult:
    if not isinstance(patches, list) or not patches:
        return _invalid(parent_code, failure_type, f"{patch_label}s must be a non-empty list")
    region_patches: list[RegionPatch] = []
    for item in patches:
        if not isinstance(item, dict):
            return _invalid(parent_code, failure_type, f"each {patch_label} must be an object")
        region = str(item.get("region") or "").strip()
        body = item.get("body")
        operation = str(item.get("operation") or "replace").strip().lower()
        if not region or body is None:
            return _invalid(parent_code, failure_type, f"{patch_label} requires region and body")
        if "anchor_name" in item:
            return _invalid(parent_code, "invalid_patch_format", "region_patches must use region, not anchor_name")
        if operation not in {"replace", "append"}:
            return _invalid(parent_code, failure_type, f"unsupported {patch_label} operation: {operation}")
        region_patches.append(RegionPatch(region=region, body=str(body), operation=operation))
    try:
        result = apply_region_patches(parent_code, region_patches)
    except Exception as exc:
        return _invalid(parent_code, "region_apply_failed", f"failed to apply region patch: {exc}")
    return CandidateNormalizeResult(code=result.code, ok=True, logs=result.logs)


def _try_parse_patch_payload(text: str) -> dict[str, Any] | None:
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict) and "region_patches" in payload:
            return payload
        if isinstance(payload, dict) and any(key in payload for key in ("anchor_patches", "anchor_name", "patches", "edits")):
            return {"_invalid_patch_payload": payload}
    return None


def _looks_like_unapplied_patch(text: str) -> bool:
    lowered = text.lower()
    return (
        "region_patches" in lowered
        or "anchor_patches" in lowered
        or "anchor_name" in lowered
        or "patches" in lowered and text.lstrip().startswith("{")
        or "edits" in lowered and text.lstrip().startswith("{")
        or "region" in lowered and text.lstrip().startswith("{")
        or text.lstrip().startswith("{")
    )


def _looks_like_python_module(text: str) -> bool:
    return "class ModelNew" in text or "def forward" in text or "CUDA_CU_SRC" in text or "CUDA_CPP_SRC" in text


def _has_editable_regions(source_code: str) -> bool:
    return "# <<<IMPROVE:" in source_code and "# <<<END_IMPROVE>>>" in source_code


def _invalid(parent_code: str, failure_type: str, message: str) -> CandidateNormalizeResult:
    return CandidateNormalizeResult(code=parent_code, ok=False, failure_type=failure_type, logs=[message])


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()
