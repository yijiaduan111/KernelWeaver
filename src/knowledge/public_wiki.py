"""Frozen public knowledge, with no benchmark identity or experiment input."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .domain_wiki import DomainWiki, _load_frontmatter
from ..config import asset_path


PUBLIC_ROOT = asset_path("knowledge/kernelwiki_public")
_LEAK = re.compile(r"\b(?:kernelbench|L\d+[_ -]?P\d+|P\d{1,3}|ModelNew)\b|\bruns/|\bbest_code\.py\b|\b\d+(?:\.\d+)?\s*[x×]\s*speedup", re.I)
_FORBIDDEN_KEYS = {"problem_ids", "problem_id", "task_name", "performance_claims", "benchmark_results", "match", "shapes", "shape", "speedup"}
_METHODS = {"sum", "mean", "amax", "amin", "cumsum", "cumprod", "matmul", "mm", "bmm", "softmax", "log_softmax", "sigmoid", "relu", "tanh"}


def corpus_files(path: Path) -> list[Path]:
    files = [path / "index.yaml"]
    for directory in ("data", "sources", "wiki"):
        base = path / directory
        if base.is_symlink():
            raise ValueError(f"Public Wiki directory cannot be a symlink: {base}")
        files.extend(p for p in base.rglob("*") if p.is_file())
    for file in files:
        if file.is_symlink() or not file.resolve().is_relative_to(path.resolve()):
            raise ValueError(f"Public Wiki file escapes its corpus: {file}")
    return sorted(files)


def corpus_manifest(path: Path) -> dict:
    hashes = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in corpus_files(path)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"version": 1, "sha256": digest, "files": hashes}


def _all_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_keys(nested)


def audit_public_corpus(path: Path, *, check_lock: bool = True) -> dict:
    """Fail closed on missing provenance, unsafe source links, or changed corpus.

    These checks detect common contamination; source review is still required.
    A removed benchmark label cannot prove that an idea was independently derived.
    """
    path = Path(path)
    metadata = yaml.safe_load((path / "index.yaml").read_text())
    if metadata.get("knowledge_scope") != "public_general":
        raise ValueError("Strict retrieval requires knowledge_scope: public_general")
    pages = {}
    for file in corpus_files(path):
        raw = file.read_text(encoding="utf-8")
        if _LEAK.search(raw):
            raise ValueError(f"Benchmark/experiment marker in public corpus: {file}")
        if file.suffix == ".md":
            parsed = _load_frontmatter(file)
            if parsed is None:
                raise ValueError(f"Public page requires frontmatter: {file}")
            front, _ = parsed
            required = {"id", "title", "type", "knowledge_scope", "origin", "status", "confidence", "reproducibility"}
            if required - front.keys():
                raise ValueError(f"Missing provenance fields in {file}: {required - front.keys()}")
            if front["id"] in pages:
                raise ValueError(f"Duplicate public page: {front['id']}")
            if (front["knowledge_scope"] != "public_general" or front["origin"] != "public_documentation"
                    or front["status"] != "reviewed" or front["reproducibility"] != "documentation"):
                raise ValueError(f"Unreviewed or empirical page in public corpus: {file}")
            if front["confidence"] not in {"source-reported", "inferred"}:
                raise ValueError(f"Unsupported evidence claim: {file}")
            if file.relative_to(path).parts[0] == "sources":
                url = urlparse(str(front.get("url", "")))
                pinned = (url.hostname == "docs.nvidia.com" and "/archive/12.6.0/" in url.path
                          or url.hostname == "raw.githubusercontent.com" and url.path.startswith("/pytorch/pytorch/v2.6.0/")
                          or url.hostname == "docs.pytorch.org" and url.path.startswith("/docs/2.6/")
                          or url.hostname == "developer.nvidia.com" and url.path == "/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda"
                          or url.hostname == "arxiv.org" and url.path == "/abs/2205.14135v1")
                if url.scheme != "https" or not pinned or not front.get("revision"):
                    raise ValueError(f"Source needs an approved, versioned primary URL: {file}")
            elif not front.get("sources"):
                raise ValueError(f"Wiki page has no sources: {file}")
            pages[front["id"]] = front
            data = front
        elif file.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        else:
            raise ValueError(f"Unexpected corpus file: {file}")
        forbidden = set(_all_keys(data)) & _FORBIDDEN_KEYS
        if forbidden:
            raise ValueError(f"Task-specific fields in {file}: {forbidden}")
    for page in pages.values():
        for source in page.get("sources", []):
            if source not in pages or pages[source]["type"] != "source":
                raise ValueError(f"Missing primary source {source} for {page['id']}")
    manifest = corpus_manifest(path)
    if check_lock:
        lock = json.loads((path / "corpus.lock.json").read_text())
        if any(lock.get(key) != value for key, value in manifest.items()):
            raise ValueError("Public Wiki changed since freeze; review and explicitly freeze a new version")
    return {"scope": "public_general", "pages": len(pages), **manifest}


def operator_symbols(source: str) -> set[str]:
    """Read only API/operator structure, excluding names, constants and fixtures."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imports = {"torch": "torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imports[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("torch"):
            for item in node.names:
                imports[item.asname or item.name] = f"{node.module}.{item.name}"

    def dotted(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{dotted(node.value)}.{node.attr}"
        return ""

    symbols = set()

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if node.name.startswith(("get_inputs", "get_init_inputs", "test_", "benchmark_")):
                return
            self.generic_visit(node)

        def visit_Call(self, node):
            name = dotted(node.func)
            leaf = name.rsplit(".", 1)[-1].lower()
            if name.startswith("torch.") or (isinstance(node.func, ast.Attribute) and leaf in _METHODS):
                symbols.add(leaf)
            self.generic_visit(node)

        def visit_BinOp(self, node):
            if isinstance(node.op, ast.MatMult):
                symbols.add("matmul")
            self.generic_visit(node)

    Visitor().visit(tree)
    return symbols


class PublicKernelWiki(DomainWiki):
    """Public-only retrieval independent of task IDs, shapes and run outcomes."""

    def __init__(self, path, *, target_arch, experimental_page_ids=None):
        if experimental_page_ids:
            raise ValueError("Public Wiki cannot opt into experimental pages")
        self.audit = audit_public_corpus(Path(path))
        super().__init__(path, target_arch=target_arch)
        self.knowledge_scope = "public_general"
        self.corpus_sha256 = self.audit["sha256"]
        self.families = yaml.safe_load((self.path / "data/operators.yaml").read_text())
        for page in self.pages:
            for family in page["frontmatter"].get("operator_families", []):
                if family not in self.families:
                    raise ValueError(f"Unknown operator family: {family}")

    def task_families(self, task) -> set[str]:
        symbols = operator_symbols(task.reference_code)
        matched = set()
        for family, rule in self.families.items():
            direct = set(rule.get("any", [])) & symbols
            composites = any(set(group) <= symbols for group in rule.get("all_groups", []))
            if direct or composites:
                matched.add(family)
        return matched

    def retrieve(self, task):
        if task.backend != self.backend or not self.max_entries:
            return []
        families = self.task_families(task)
        candidates, fallback, hardware = [], [], []
        for page in self.pages:
            f = page["frontmatter"]
            if not page["path"].startswith("wiki/"):
                continue
            if self.target_arch not in f.get("architectures", []) or self.backend not in f.get("backends", []):
                continue
            overlap = families & set(f.get("operator_families", []))
            if overlap:
                score = int(f.get("priority", 0)) + self._score_page(f, page["body"], sorted(overlap))
                candidates.append((score, page["path"], page))
            elif f.get("type") == "hardware":
                hardware.append(page)
            elif f.get("type") == "contract":
                fallback.append(page)
        candidates.sort(key=lambda x: (-x[0], x[1]))
        # Reserve one context slot for hardware; unmatched operators get only
        # a contract checklist, with the missing semantic coverage made explicit.
        selected = [(page, score) for score, _, page in candidates[:max(1, self.max_entries - 1)]]
        if not selected:
            selected = [(page, 0) for page in fallback[:1]]
        selected += [(page, 0) for page in hardware[:1]]
        return [self._page_evidence(page, score, sorted(families)) for page, score in selected[:self.max_entries]]

    def _page_evidence(self, page, score, query_terms):
        result = super()._page_evidence(page, score, query_terms)
        f = page["frontmatter"]
        sources = {p["frontmatter"]["id"]: p["frontmatter"] for p in self.pages if p["frontmatter"]["type"] == "source"}
        result.update({
            "knowledge_scope": "public_general", "origin": f["origin"],
            "corpus_sha256": self.corpus_sha256,
            "applicability": f.get("applicability", []),
            "operator_families": f.get("operator_families", []),
            "coverage": "general_contract_only" if f["type"] == "contract" else f["type"],
            "source_records": [{"id": sid, "url": sources[sid]["url"], "revision": sources[sid]["revision"]} for sid in f.get("sources", [])],
        })
        return result
