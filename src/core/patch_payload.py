"""Shared helpers for extracting and canonicalizing patch payloads."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable


def parse_loose_json_dict(text: str, *, allow_python_literal: bool = True) -> dict[str, Any] | None:
    for candidate in iter_json_object_candidates(text):
        for parser in _candidate_parsers(allow_python_literal=allow_python_literal):
            try:
                payload = parser(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    return None


def iter_json_object_candidates(text: str) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    candidates: list[str] = []
    _append_unique(candidates, cleaned)
    for candidate in extract_balanced_json_objects(cleaned):
        _append_unique(candidates, candidate)
    return candidates


def extract_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start_index: int | None = None
    depth = 0
    quote_char: str | None = None
    escaping = False
    for index, char in enumerate(str(text or "")):
        if start_index is None:
            if char == "{":
                start_index = index
                depth = 1
                quote_char = None
                escaping = False
            continue
        if quote_char is not None:
            if escaping:
                escaping = False
                continue
            if char == "\\":
                escaping = True
                continue
            if char == quote_char:
                quote_char = None
            continue
        if char in {'"', "'"}:
            quote_char = char
            continue
        if char == "{":
            depth += 1
            continue
        if char != "}":
            continue
        depth -= 1
        if depth == 0:
            objects.append(text[start_index : index + 1])
            start_index = None
    return objects


def canonicalize_region_patches(payload: dict[str, Any], allowed_regions: dict[str, str]) -> list[dict[str, str]] | None:
    patches = payload.get("region_patches")
    if not isinstance(patches, list):
        for key in ("patches", "edits", "anchor_patches"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                patches = candidate
                break
    if not isinstance(patches, list) or not patches:
        return None

    normalized: list[dict[str, str]] = []
    for item in patches:
        if not isinstance(item, dict):
            return None
        region = str(item.get("region") or item.get("anchor_name") or item.get("anchor") or "").strip()
        if not region:
            return None
        if allowed_regions and region not in allowed_regions:
            return None
        body = item.get("body")
        if body is None:
            body = item.get("code")
        if body is None:
            body = item.get("replacement")
        if body is None:
            body = item.get("new_body")
        if body is None:
            body = item.get("content")
        if not isinstance(body, str) or not body.strip():
            return None
        operation = str(item.get("operation") or allowed_regions.get(region, "replace")).strip().lower()
        if operation not in {"replace", "append"}:
            operation = allowed_regions.get(region, "replace")
        normalized.append(
            {
                "region": region,
                "operation": operation,
                "body": strip_region_marker_lines(body),
            }
        )
    return normalized or None


def strip_region_marker_lines(body: str) -> str:
    lines: list[str] = []
    for line in str(body).strip().splitlines():
        if re.search(r"#\s*<<<(?:END_)?IMPROVE", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip("\n")


def _append_unique(items: list[str], candidate: str) -> None:
    text = str(candidate or "").strip()
    if text and text not in items:
        items.append(text)


def _candidate_parsers(*, allow_python_literal: bool) -> tuple[Callable[[str], Any], ...]:
    if not allow_python_literal:
        return (json.loads,)
    return (json.loads, ast.literal_eval)
