"""Editable region helpers for generated KernelWeaver candidates.

The LLM edits region bodies only. The surrounding scaffold and marker comments
remain owned by deterministic code so candidates cannot accidentally rewrite the
runtime contract.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass

from .hygiene import normalize_region_body


_REGION_PATTERN = re.compile(
    r"(?ms)(^[ \t]*#\s*<<<IMPROVE:(?P<name>[^>]+)>>>(?:\r?\n))(?P<body>.*?)(^[ \t]*#\s*<<<END_IMPROVE>>>)"
)


@dataclass(frozen=True)
class RegionPatch:
    region: str
    body: str
    operation: str = "replace"


@dataclass(frozen=True)
class RegionApplyResult:
    code: str
    logs: list[str]


def extract_region_names(source_code: str) -> list[str]:
    """Return editable region names in scaffold order."""
    return [match.group("name").strip() for match in _REGION_PATTERN.finditer(source_code)]


def region_exists(source_code: str, region: str) -> bool:
    return region in extract_region_names(source_code)


def apply_region_patches(
    source_code: str,
    patches: list[RegionPatch],
    *,
    allowed_regions: set[str] | None = None,
    frozen_regions: set[str] | None = None,
) -> RegionApplyResult:
    """Apply region-body patches while preserving all marker comments."""
    if not patches:
        raise ValueError("region_patches must be a non-empty list")
    available = extract_region_names(source_code)
    if len(available) != len(set(available)):
        raise ValueError(f"duplicate editable region names in scaffold: {available}")
    code = source_code
    logs: list[str] = []
    seen: set[str] = set()
    for patch in patches:
        name = patch.region.strip()
        if not name:
            raise ValueError("region patch requires region")
        if name in seen:
            raise ValueError(f"duplicate patch for region: {name}")
        seen.add(name)
        if name not in available:
            raise ValueError(f"editable region '{name}' not found; available={available}")
        if allowed_regions is not None and name not in allowed_regions:
            raise ValueError(f"region '{name}' is not active in current refinement policy")
        if frozen_regions is not None and name in frozen_regions:
            raise ValueError(f"region '{name}' is frozen in current refinement policy")
        operation = patch.operation.strip().lower() or "replace"
        if operation not in {"replace", "append"}:
            raise ValueError(f"unsupported region operation for {name}: {operation}")
        body = _strip_region_markers(patch.body)
        hygiene = normalize_region_body(name, body)
        code = _apply_single_region(code, name, hygiene.body, operation)
        logs.extend(hygiene.logs)
        logs.append(f"applied_region_patch:{name}:{operation}")
    if not preserve_region_scaffold(source_code, code):
        raise ValueError("region patch changed protected scaffold")
    return RegionApplyResult(code=code, logs=logs)


def preserve_region_scaffold(original_code: str, candidate_code: str) -> bool:
    """Return True when only editable region bodies changed."""
    return _split_region_scaffold(original_code) == _split_region_scaffold(candidate_code)


def validate_region_scaffold(source_code: str) -> tuple[bool, str | None]:
    names = extract_region_names(source_code)
    if not names:
        return False, "candidate has no editable region markers"
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        return False, f"duplicate editable region markers: {duplicates}"
    return True, None


def _apply_single_region(source_code: str, region: str, new_body: str, operation: str) -> str:
    match = None
    for candidate in _REGION_PATTERN.finditer(source_code):
        if candidate.group("name").strip() == region:
            match = candidate
            break
    if match is None:
        raise ValueError(f"editable region '{region}' not found")
    indent = re.match(r"^([ \t]*)", match.group(1)).group(1)
    current_body = textwrap.dedent(match.group("body")).strip("\n")
    incoming_body = textwrap.dedent(new_body).strip("\n")
    logical_body = incoming_body if operation == "replace" else "\n".join(piece for piece in [current_body, incoming_body] if piece)
    if logical_body:
        rendered_body = "\n".join(f"{indent}{line}" if line else "" for line in logical_body.splitlines()) + "\n"
    else:
        rendered_body = ""
    return f"{source_code[:match.start()]}{match.group(1)}{rendered_body}{match.group(4)}{source_code[match.end():]}"


def _strip_region_markers(body: str) -> str:
    lines = []
    for line in str(body).strip().splitlines():
        if re.search(r"#\s*<<<(?:END_)?IMPROVE", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip("\n")


def _split_region_scaffold(source_code: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in _REGION_PATTERN.finditer(source_code):
        parts.append(("text", source_code[cursor:match.start()]))
        parts.append(("region", match.group("name").strip()))
        parts.append(("begin", match.group(1)))
        parts.append(("end", match.group(4)))
        cursor = match.end()
    parts.append(("text", source_code[cursor:]))
    return parts
