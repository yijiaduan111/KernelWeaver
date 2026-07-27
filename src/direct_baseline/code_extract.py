"""Minimal code extraction for direct LLM responses."""

from __future__ import annotations

import re
import textwrap


_PYTHON_FENCE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_code(response_text: str) -> str:
    """Extract a complete Python source candidate from an LLM response.

    This intentionally does not repair or rewrite code. The direct baseline is
    supposed to measure the model's one-shot output, not a guarded framework.
    """
    text = (response_text or "").strip()
    if not text:
        return ""

    fenced_blocks = [match.group("code") for match in _PYTHON_FENCE.finditer(text)]
    if fenced_blocks:
        selected = next((block for block in fenced_blocks if "class ModelNew" in block), fenced_blocks[0])
        return _normalize_source(selected)
    return _normalize_source(text)


def _normalize_source(source: str) -> str:
    normalized = textwrap.dedent(source).strip()
    return f"{normalized}\n" if normalized else ""
