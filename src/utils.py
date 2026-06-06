from __future__ import annotations

import math
import re
import textwrap
import hashlib
from typing import Any


_ANCHOR_PATTERN = re.compile(
    r"(?ms)(^[ \t]*#\s*<<<IMPROVE:(?P<name>[^>]+)>>>(?:\r?\n))(?P<body>.*?)(^[ \t]*#\s*<<<END_IMPROVE>>>)"
)


def extract_anchor_names(source_code: str) -> list[str]:
    return re.findall(r"(?m)^[ 	]*#\s*<<<IMPROVE:([^>]+)>>>", source_code)


def preserve_anchor_scaffold(original_code: str, candidate_code: str) -> bool:
    """Return True when only anchor bodies changed.

    This enforces that anchor names, marker order, and all non-anchor
    scaffold text remain identical between two versions of the same file.
    """
    original_parts = _split_anchor_scaffold(original_code)
    candidate_parts = _split_anchor_scaffold(candidate_code)
    return original_parts == candidate_parts


def apply_anchor_edit(source_code: str, anchor_name: str, new_body: str, operation: str = "replace") -> str:
    match = None
    for candidate in _ANCHOR_PATTERN.finditer(source_code):
        if candidate.group("name") == anchor_name:
            match = candidate
            break
    if match is None:
        raise ValueError(f"Anchor '{anchor_name}' not found in source.")
    if operation not in {"replace", "append"}:
        raise ValueError(f"Unsupported anchor operation: {operation}")

    indent = re.match(r"^([ \t]*)", match.group(1)).group(1)
    current_body = textwrap.dedent(match.group("body")).strip("\n")
    incoming_body = textwrap.dedent(new_body).strip("\n")
    if operation == "replace":
        logical_body = incoming_body
    else:
        pieces = [piece for piece in [current_body, incoming_body] if piece]
        logical_body = "\n".join(pieces)
    if logical_body:
        lines = logical_body.splitlines()
        rendered_body = "\n".join(f"{indent}{line}" if line else "" for line in lines) + "\n"
    else:
        rendered_body = ""
    return (
        f"{source_code[:match.start()]}"
        f"{match.group(1)}{rendered_body}{match.group(4)}"
        f"{source_code[match.end():]}"
    )


def replace_anchor_body(source_code: str, anchor_name: str, new_body: str) -> str:
    return apply_anchor_edit(source_code, anchor_name, new_body, operation="replace")


def compare_values(actual: Any, expected: Any, tolerance: float = 1e-5) -> bool:
    torch = _maybe_import_torch()
    if torch is not None and isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if actual.shape != expected.shape:
            return False
        if actual.dtype != expected.dtype:
            expected = expected.to(dtype=actual.dtype)
        return bool(torch.allclose(actual, expected.to(device=actual.device), rtol=tolerance, atol=tolerance))
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        return all(compare_values(left, right, tolerance) for left, right in zip(actual, expected))
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        if len(actual) != len(expected):
            return False
        return all(compare_values(left, right, tolerance) for left, right in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return False
        return all(compare_values(actual[key], expected[key], tolerance) for key in actual)
    return actual == expected


def shorten_runtime(runtime: float | None) -> str:
    if runtime is None:
        return "n/a"
    return f"{runtime * 1_000_000:.1f} us"


def clone_value(value: Any) -> Any:
    torch = _maybe_import_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def normalized_code_hash(source_code: str) -> str:
    normalized = "\n".join(line.rstrip() for line in source_code.strip().splitlines())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def last_log_excerpt(logs: list[str], max_length: int = 160) -> str | None:
    if not logs:
        return None
    excerpt = str(logs[-1]).strip()
    if len(excerpt) <= max_length:
        return excerpt
    return excerpt[: max_length - 3] + "..."


def _split_anchor_scaffold(source_code: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in _ANCHOR_PATTERN.finditer(source_code):
        parts.append(("text", source_code[cursor:match.start()]))
        parts.append(("anchor", str(match.group("name"))))
        parts.append(("begin", str(match.group(1))))
        parts.append(("end", str(match.group(4))))
        cursor = match.end()
    parts.append(("text", source_code[cursor:]))
    return parts


def _maybe_import_torch():
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch
