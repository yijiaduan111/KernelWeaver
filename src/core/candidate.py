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

from ..utils import apply_anchor_edit


@dataclass
class CandidateNormalizeResult:
    code: str
    ok: bool
    failure_type: str | None = None
    logs: list[str] = field(default_factory=list)


def normalize_candidate(parent_code: str, raw_output: str) -> CandidateNormalizeResult:
    """Return source code from full-file output or JSON anchor patches."""
    text = _strip_code_fences(str(raw_output or "")).strip()
    if not text:
        return _invalid(parent_code, "empty_candidate", "candidate output is empty")

    patch_payload = _try_parse_patch_payload(text)
    if patch_payload is not None:
        return _apply_patch_payload(parent_code, patch_payload)

    if _looks_like_unapplied_patch(text):
        return _invalid(parent_code, "invalid_patch_format", "candidate looks like anchor patch JSON but could not be parsed")

    if _looks_like_python_module(text):
        return CandidateNormalizeResult(code=text, ok=True)

    return _invalid(parent_code, "invalid_candidate_format", "candidate is neither Python source nor valid anchor patch JSON")


def _apply_patch_payload(parent_code: str, payload: dict[str, Any]) -> CandidateNormalizeResult:
    patches = payload.get("anchor_patches")
    if not isinstance(patches, list) or not patches:
        return _invalid(parent_code, "invalid_patch_format", "anchor_patches must be a non-empty list")
    code = parent_code
    logs: list[str] = []
    try:
        for item in patches:
            if not isinstance(item, dict):
                return _invalid(parent_code, "invalid_patch_format", "each anchor patch must be an object")
            name = str(item.get("anchor_name") or "").strip()
            body = item.get("body")
            operation = str(item.get("operation") or "replace").strip()
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
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict) and "anchor_patches" in payload:
            return payload
    return None


def _looks_like_unapplied_patch(text: str) -> bool:
    lowered = text.lower()
    return "anchor_patches" in lowered or "anchor_name" in lowered or text.lstrip().startswith("{")


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
