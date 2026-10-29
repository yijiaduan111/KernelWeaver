"""Architecture-scoped project Wiki with KDA-style deterministic retrieval."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..models import TaskSpec


_FRONTMATTER = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", re.DOTALL)
_SEARCH_FIELDS = (
    "tags", "techniques", "hardware_features", "kernel_types", "languages",
    "aliases", "symptoms", "op_types",
)
_TASK_STOPWORDS = {
    "kernelbench", "level", "problem", "model", "simple", "tensor", "torch",
    "cuda", "forward", "input", "output", "over", "with", "and", "the",
}


class DomainWiki:
    """Retrieve bounded evidence from either a legacy YAML file or a Wiki corpus.

    A corpus follows KernelWiki's public shape: Markdown pages with YAML
    frontmatter, an alias registry, source links, and deterministic weighted
    search. Applicability constraints remain hard gates so a high lexical score
    cannot leak advice across architectures or task contracts.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        target_arch: str,
        experimental_page_ids: list[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.corpus_mode = self.path.is_dir()
        metadata_path = self.path / "index.yaml" if self.corpus_mode else self.path
        payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Domain Wiki must contain a mapping: {metadata_path}")
        architectures = {_normal(item) for item in payload.get("architectures") or []}
        if _normal(target_arch) not in architectures:
            raise ValueError(f"Domain Wiki {self.path} does not support {target_arch}")
        self.target_arch = _normal(target_arch)
        self.experimental_page_ids = {
            str(page_id).strip() for page_id in (experimental_page_ids or []) if str(page_id).strip()
        }
        self.backend = _normal(payload.get("backend") or "")
        policy = payload.get("policy") or {}
        self.max_entries = max(0, int(policy.get("max_entries", 1)))
        self.max_body_chars = max(0, int(policy.get("max_body_chars", 1400)))
        self.global_constraints = [str(item) for item in policy.get("global_constraints") or []]
        self.entries = [entry for entry in payload.get("entries") or [] if isinstance(entry, dict)]
        self.alias_to_canonical: dict[str, str] = {}
        self.pages: list[dict[str, Any]] = []
        if self.corpus_mode:
            self.alias_to_canonical = self._load_aliases()
            self.pages = self._load_pages()

    def retrieve(self, task: TaskSpec) -> list[dict[str, Any]]:
        if self.max_entries <= 0 or _normal(task.backend or "") != self.backend:
            return []
        if self.corpus_mode:
            return self._retrieve_corpus(task)
        return self._retrieve_legacy(task)

    def query(
        self,
        keywords: str | list[str],
        *,
        page_type: str | None = None,
        tag: str | None = None,
        language: str | None = None,
        architecture: str | None = None,
        symptom: str | None = None,
        confidence: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Expose KernelWiki-style free-text search for agents and humans."""
        if not self.corpus_mode:
            raise ValueError("Free-text query requires a directory-backed Wiki corpus")
        raw = [keywords] if isinstance(keywords, str) else keywords
        query_terms = [
            _normal(token)
            for item in raw
            for token in re.split(r"\s+", str(item).strip())
            if token
        ]
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for page in self.pages:
            frontmatter = page["frontmatter"]
            actual_type = _normal(frontmatter.get("type") or "source")
            if page_type and actual_type != _normal(page_type):
                continue
            architectures = {_normal(item) for item in frontmatter.get("architectures") or []}
            requested_arch = _normal(architecture or self.target_arch)
            if architectures and requested_arch not in architectures:
                continue
            if confidence and _normal(frontmatter.get("confidence") or "") != _normal(confidence):
                continue
            if language and _normal(language) not in {_normal(item) for item in frontmatter.get("languages") or []}:
                continue
            if symptom and _normal(symptom) not in {_normal(item) for item in frontmatter.get("symptoms") or []}:
                continue
            if tag:
                available = {
                    _normal(value)
                    for key in _SEARCH_FIELDS
                    for value in (frontmatter.get(key) or [])
                }
                variants = {_normal(tag)}
                canonical = self.alias_to_canonical.get(_normal(tag))
                if canonical:
                    variants.add(_normal(canonical))
                if not available.intersection(variants):
                    continue
            score = self._score_page(frontmatter, page["body"], query_terms) if query_terms else 1
            if score > 0:
                ranked.append((score, page["path"], page))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [self._page_evidence(page, score, query_terms) for score, _path, page in ranked[:max(0, limit)]]

    def _retrieve_legacy(self, task: TaskSpec) -> list[dict[str, Any]]:
        op_type = _task_op_type(task)
        source = task.reference_code.lower()
        matches: list[dict[str, Any]] = []
        for entry in self.entries:
            if not self._rule_applies(entry.get("match") or {}, task, op_type, source):
                continue
            matches.append({
                "id": str(entry["id"]),
                "title": str(entry.get("title") or entry["id"]),
                "target_arch": self.target_arch,
                "global_constraints": list(self.global_constraints),
                "advice": _strings(entry.get("advice")),
                "avoid": _strings(entry.get("avoid")),
                "acceptance": _strings(entry.get("acceptance")),
                "sources": _strings(entry.get("sources")),
            })
            if len(matches) >= self.max_entries:
                break
        return matches

    def _retrieve_corpus(self, task: TaskSpec) -> list[dict[str, Any]]:
        op_type = _task_op_type(task)
        source = task.reference_code.lower()
        query_terms = self._task_query_terms(task)
        ranked_wiki: list[tuple[int, str, dict[str, Any]]] = []
        ranked_sources: list[tuple[int, str, dict[str, Any]]] = []
        for page in self.pages:
            frontmatter = page["frontmatter"]
            # Human/free-text queries can inspect draft and rejected pages, but
            # agents only receive entries that passed the matched A/B gate.
            if page["path"].startswith("wiki/"):
                status = _normal(frontmatter.get("status") or "draft")
                page_id = str(frontmatter.get("id") or "")
                # Rejected pages are never sent to agents.  A named draft can
                # be tested by an isolated profile without globally promoting
                # unreplicated guidance.
                if status != "active" and not (
                    status == "draft" and page_id in self.experimental_page_ids
                ):
                    continue
            page_backends = {_normal(item) for item in frontmatter.get("backends") or []}
            if page_backends and self.backend not in page_backends:
                continue
            page_architectures = {_normal(item) for item in frontmatter.get("architectures") or []}
            if page_architectures and self.target_arch not in page_architectures:
                continue
            rule = dict(frontmatter.get("match") or {})
            if "op_types" not in rule and frontmatter.get("op_types"):
                rule["op_types"] = frontmatter["op_types"]
            if not self._rule_applies(rule, task, op_type, source):
                continue
            score = self._score_page(frontmatter, page["body"], query_terms)
            if score > 0:
                ranked = ranked_wiki if page["path"].startswith("wiki/") else ranked_sources
                ranked.append((score, page["path"], page))
        ranked_wiki.sort(key=lambda item: (-item[0], item[1]))
        ranked_sources.sort(key=lambda item: (-item[0], item[1]))
        selected = ranked_wiki[:self.max_entries]
        if selected and len(selected) < self.max_entries:
            source_ids = {
                str(source_id)
                for _score, _path, page in selected
                for source_id in (page["frontmatter"].get("sources") or [])
            }
            selected.extend(
                item for item in ranked_sources
                if str(item[2]["frontmatter"].get("id")) in source_ids
            )
            selected = selected[:self.max_entries]
        return [self._page_evidence(page, score, query_terms) for score, _path, page in selected]

    def _rule_applies(
        self,
        rule: dict[str, Any],
        task: TaskSpec,
        op_type: str,
        source: str,
    ) -> bool:
        op_types = {_normal(item) for item in rule.get("op_types") or []}
        problem_ids = {int(item) for item in rule.get("problem_ids") or []}
        source_any = [_normal(item) for item in rule.get("source_any") or []]
        source_all = [_normal(item) for item in rule.get("source_all") or []]
        source_none = [_normal(item) for item in rule.get("source_none") or []]
        if op_types and op_type not in op_types:
            return False
        if problem_ids and task.problem_id not in problem_ids:
            return False
        if source_any and not any(token in source for token in source_any):
            return False
        if source_all and not all(token in source for token in source_all):
            return False
        if source_none and any(token in source for token in source_none):
            return False
        return True

    def _load_aliases(self) -> dict[str, str]:
        aliases_path = self.path / "data" / "aliases.yaml"
        if not aliases_path.exists():
            return {}
        payload = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Wiki aliases must contain a mapping: {aliases_path}")
        aliases: dict[str, str] = {}
        for canonical, variants in payload.items():
            if not isinstance(canonical, str):
                continue
            aliases.setdefault(_normal(canonical), str(canonical))
            for variant in variants or []:
                if isinstance(variant, str):
                    aliases.setdefault(_normal(variant), str(canonical))
        return aliases

    def _load_pages(self) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for directory in ("sources", "wiki"):
            base = self.path / directory
            if not base.exists():
                continue
            for markdown in sorted(base.rglob("*.md")):
                parsed = _load_frontmatter(markdown)
                if parsed is None:
                    continue
                frontmatter, body = parsed
                page_id = str(frontmatter.get("id") or "")
                if not page_id:
                    raise ValueError(f"Wiki page is missing id: {markdown}")
                if page_id in seen_ids:
                    raise ValueError(f"Duplicate Wiki page id {page_id}: {markdown}")
                seen_ids.add(page_id)
                pages.append({
                    "path": str(markdown.relative_to(self.path)),
                    "frontmatter": frontmatter,
                    "body": body,
                })
        return pages

    def _task_query_terms(self, task: TaskSpec) -> list[str]:
        op_type = _task_op_type(task)
        haystack = "\n".join((task.name, task.description, task.reference_code)).lower()
        terms: list[str] = []
        _append_unique(terms, op_type)

        # Like KernelWiki, normalize known vocabulary through a central alias
        # registry before ranking pages.
        for alias, canonical in self.alias_to_canonical.items():
            if alias and alias in haystack:
                _append_unique(terms, _normal(canonical))

        for call in re.findall(r"(?:torch|nn)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+", haystack):
            _append_unique(terms, _normal(call))
            _append_unique(terms, _normal(call.rsplit(".", 1)[-1]))

        name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", task.name).lower()
        for token in re.findall(r"[a-z][a-z0-9_]+", name):
            normalized = _normal(token)
            if normalized not in _TASK_STOPWORDS and not normalized.startswith("l1"):
                _append_unique(terms, normalized)
        return terms[:20]

    def _score_page(self, frontmatter: dict[str, Any], body: str, keywords: list[str]) -> int:
        title = _normal(frontmatter.get("title") or "")
        tag_text = " ".join(
            str(value)
            for key in _SEARCH_FIELDS
            for value in (frontmatter.get(key) or [])
        ).lower()
        body_lower = body.lower()
        score = 0
        for keyword in keywords:
            variants = [keyword]
            canonical = self.alias_to_canonical.get(_normal(keyword))
            if canonical and _normal(canonical) != _normal(keyword):
                variants.append(_normal(canonical))
            best = 0
            for variant in variants:
                candidate = 0
                if variant in title:
                    candidate += 10
                if variant in tag_text:
                    candidate += 5
                candidate += min(body_lower.count(variant), 3)
                best = max(best, candidate)
            score += best
        return score

    def _page_evidence(
        self,
        page: dict[str, Any],
        score: int,
        query_terms: list[str],
    ) -> dict[str, Any]:
        frontmatter = page["frontmatter"]
        body = page["body"].strip()
        if self.max_body_chars and len(body) > self.max_body_chars:
            body = body[:self.max_body_chars].rsplit(" ", 1)[0].rstrip() + "…"
        return {
            "id": str(frontmatter["id"]),
            "title": str(frontmatter.get("title") or frontmatter["id"]),
            "page_type": str(frontmatter.get("type") or "source"),
            "path": page["path"],
            "target_arch": self.target_arch,
            "score": score,
            "query_terms": list(query_terms),
            "confidence": str(frontmatter.get("confidence") or "experimental"),
            "reproducibility": str(frontmatter.get("reproducibility") or "description"),
            "status": str(frontmatter.get("status") or "active"),
            "tags": _strings(frontmatter.get("tags")),
            "global_constraints": list(self.global_constraints),
            "advice": _strings(frontmatter.get("advice")),
            "avoid": _strings(frontmatter.get("avoid")),
            "acceptance": _strings(frontmatter.get("acceptance")),
            "sources": _strings(frontmatter.get("sources")),
            "performance_claims": list(frontmatter.get("performance_claims") or []),
            "content": body,
        }


def _load_frontmatter(path: Path) -> tuple[dict[str, Any], str] | None:
    match = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        return None
    frontmatter = yaml.safe_load(match.group(1))
    if not isinstance(frontmatter, dict):
        return None
    return frontmatter, match.group(2)


def _task_op_type(task: TaskSpec) -> str:
    return _normal(task.semantic_profile.op_type if task.semantic_profile else "unknown")


def _normal(value: Any) -> str:
    return str(value).strip().lower()


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value or []]


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)
