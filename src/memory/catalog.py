from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_RULEBOOK_PATH = Path(__file__).resolve().parents[1] / "km_bottleneck.yaml"
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class MethodCatalogEntry:
    method_id: str
    title: str
    summary: str
    mechanism_requirements: tuple[str, ...]
    expected_metric_change: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]
    source: str


_CACHE: dict[Path, dict[str, MethodCatalogEntry]] = {}


def load_method_catalog(yaml_path: Path | None = None) -> dict[str, MethodCatalogEntry]:
    resolved = (yaml_path or DEFAULT_RULEBOOK_PATH).resolve()
    cached = _CACHE.get(resolved)
    if cached is not None:
        return cached
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    method_catalog = ((payload.get("llm_assist") or {}).get("method_catalog") or {})
    catalog: dict[str, MethodCatalogEntry] = {}
    for method_id, item in method_catalog.items():
        if not isinstance(item, dict):
            continue
        catalog[str(method_id)] = MethodCatalogEntry(
            method_id=str(method_id),
            title=_humanize_method_id(str(method_id)),
            summary=_normalize_text(item.get("intent") or ""),
            mechanism_requirements=tuple(_string_list(item.get("mechanism_requirements"))[:6]),
            expected_metric_change=tuple(_string_list(item.get("expected_metric_change"))[:6]),
            forbidden_patterns=tuple(_string_list(item.get("forbidden_patterns"))[:6]),
            source=resolved.name,
        )
    _CACHE[resolved] = catalog
    return catalog


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _normalize_text(value)
        return [text] if text else []
    output: list[str] = []
    for item in value:
        text = _normalize_text(item)
        if text:
            output.append(text)
    return output


def _normalize_text(value) -> str:
    text = _SPACE_RE.sub(" ", str(value or "")).strip()
    return text


def _humanize_method_id(value: str) -> str:
    pieces = value.replace("_", " ").split()
    return " ".join(piece if piece.isupper() else piece.capitalize() for piece in pieces)
