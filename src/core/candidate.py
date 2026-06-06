"""Candidate normalization before evaluation.

The workflow accepts model output, not trusted source code. This module turns
supported candidate formats into a Python source file while rejecting obvious
protocol mistakes before they reach the expensive KernelBench evaluator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..utils import apply_anchor_edit
from .patch_payload import parse_loose_json_dict
from .regions import RegionPatch, apply_region_patches


@dataclass
class CandidateNormalizeResult:
    code: str
    ok: bool
    failure_type: str | None = None
    logs: list[str] = field(default_factory=list)


def normalize_candidate(parent_code: str, raw_output: str) -> CandidateNormalizeResult:
    """Return source code from JSON region/anchor patches or full-file output."""
    text = _strip_code_fences(str(raw_output or "")).strip()
    if not text:
        return _invalid(parent_code, "empty_candidate", "candidate output is empty")

    patch_payload = _try_parse_patch_payload(text)
    if patch_payload is not None:
        return _apply_patch_payload(parent_code, patch_payload)

    if _looks_like_unapplied_patch(text):
        return _invalid(parent_code, "invalid_patch_format", "candidate looks like patch JSON but could not be parsed")

    if _looks_like_python_module(text):
        return CandidateNormalizeResult(code=text, ok=True)

    return _invalid(parent_code, "invalid_candidate_format", "candidate is neither Python source nor valid patch JSON")


def _apply_patch_payload(parent_code: str, payload: dict[str, Any]) -> CandidateNormalizeResult:
    if "region_patches" in payload:
        return _apply_region_patch_payload(parent_code, payload)
    return _apply_anchor_patch_payload(parent_code, payload)


def _apply_region_patch_payload(parent_code: str, payload: dict[str, Any]) -> CandidateNormalizeResult:
    patches = payload.get("region_patches")
    if not isinstance(patches, list) or not patches:
        return _invalid(parent_code, "invalid_region_patch_format", "region_patches must be a non-empty list")
    region_patches: list[RegionPatch] = []
    for item in patches:
        if not isinstance(item, dict):
            return _invalid(parent_code, "invalid_region_patch_format", "each region patch must be an object")
        region = str(item.get("region") or item.get("anchor_name") or "").strip()
        body = item.get("body")
        operation = str(item.get("operation") or "replace").strip().lower()
        if not region or body is None:
            return _invalid(parent_code, "invalid_region_patch_format", "region patch requires region and body")
        if operation not in {"replace", "append"}:
            return _invalid(parent_code, "invalid_region_patch_format", f"unsupported region patch operation: {operation}")
        region_patches.append(RegionPatch(region=region, body=str(body), operation=operation))
    try:
        result = apply_region_patches(parent_code, region_patches)
    except Exception as exc:
        return _invalid(parent_code, "region_apply_failed", f"failed to apply region patch: {exc}")
    return CandidateNormalizeResult(code=result.code, ok=True, logs=result.logs)


def _apply_anchor_patch_payload(parent_code: str, payload: dict[str, Any]) -> CandidateNormalizeResult:
    patches = payload.get("anchor_patches")
    if not isinstance(patches, list) or not patches:
        return _invalid(parent_code, "invalid_patch_format", "anchor_patches must be a non-empty list")
    code = parent_code
    logs: list[str] = []
    try:
        for item in patches:
            if not isinstance(item, dict):
                return _invalid(parent_code, "invalid_patch_format", "each anchor patch must be an object")
            name = str(item.get("anchor_name") or item.get("region") or "").strip()
            body = item.get("body")
            operation = str(item.get("operation") or "replace").strip().lower()
            if not name or body is None:
                return _invalid(parent_code, "invalid_patch_format", "anchor patch requires anchor_name and body")
            if operation not in {"replace", "append"}:
                return _invalid(parent_code, "invalid_patch_format", f"unsupported anchor patch operation: {operation}")
            code = apply_anchor_edit(code, name, str(body), operation=operation)
            logs.append(f"applied_anchor_patch:{name}:{operation}")
    except Exception as exc:
        return _invalid(parent_code, "anchor_apply_failed", f"failed to apply anchor patch: {exc}")
    return CandidateNormalizeResult(code=code, ok=True, logs=logs)


def _try_parse_patch_payload(text: str) -> dict[str, Any] | None:
    payload = parse_loose_json_dict(text, allow_python_literal=True)
    if not isinstance(payload, dict):
        return None
    if "region_patches" in payload or "anchor_patches" in payload:
        return payload
    patches = payload.get("patches") or payload.get("edits")
    if isinstance(patches, list) and patches:
        return {"region_patches": patches}
    if payload.get("anchor_name") or payload.get("region"):
        return {"anchor_patches": [payload]}
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


def _invalid(parent_code: str, failure_type: str, message: str) -> CandidateNormalizeResult:
    return CandidateNormalizeResult(code=parent_code, ok=False, failure_type=failure_type, logs=[message])


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()
