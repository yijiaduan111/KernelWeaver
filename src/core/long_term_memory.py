"""Compact persistent experience store for cross-task kernel search."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _tokens(value: str | None) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(x) > 2}


def _op_type(task) -> str:
    profile = getattr(task, "semantic_profile", None)
    return str(getattr(profile, "op_type", "unknown") or "unknown")


def _lesson(node) -> str:
    strategy = node.plan_strategy_name or "the attempted strategy"
    failure = str(node.failure_type or "failure")
    logs = " ".join(str(x) for x in (node.logs or []) if x)
    if node.correct and node.speedup is not None and node.speedup >= 1.0:
        return f"REUSE cautiously: {strategy} was correct and reached {node.speedup:.3f}x. Preserve its validated launch and indexing structure."
    if node.correct:
        return f"CORRECT_BUT_SLOW: {strategy} was correct but not faster. Keep semantics and change one performance dimension at a time."
    if node.latest_failure_stage == "compile":
        return f"AVOID: {strategy} did not compile ({failure}). Check CUDA APIs, includes, types, launch dimensions, and wrapper signatures. {logs[-240:]}"
    if node.latest_failure_stage == "runtime":
        return f"AVOID: {strategy} failed at runtime ({failure}). Add bounds checks, 64-bit index arithmetic, valid launch geometry, and synchronization checks. {logs[-240:]}"
    if node.latest_failure_stage == "correctness":
        return f"AVOID: {strategy} compiled but was incorrect ({failure}). Recheck shape, dtype, reduction, masking, and numerical semantics. {logs[-240:]}"
    return f"AVOID: {strategy} produced an unclassified failure ({failure}). Validate before reuse."


def _memory_scope(entry: dict[str, Any]) -> str:
    """Return the portability scope of an experience.

    Backend-tagged entries are implementation-specific.  Entries explicitly
    marked generic may be shared across CUDA/NPU/ROCm, but only as abstract
    guidance rather than as code-reuse evidence.
    """
    scope = str(entry.get("memory_scope") or "").strip().lower()
    if scope in {"generic", "structural", "backend_specific"}:
        return scope
    return "backend_specific" if entry.get("backend") else "generic"


class TaskExperienceJournal:
    """Untrusted JSON journal of compact successes and failures across tasks.

    Entries are useful search hints, but are not curated operator knowledge.
    Promotion into the Wiki-backed OperatorMemory is deliberately manual.
    """

    def __init__(self, path: str | Path | None, top_k: int = 4):
        self.path = Path(path) if path else None
        self.top_k = max(1, int(top_k))
        self.entries: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                self.entries = list(payload.get("entries") or [])
            except (OSError, ValueError, TypeError):
                self.entries = []

    def retrieve(self, task) -> list[dict[str, Any]]:
        if not self.entries:
            return []
        task_tokens = _tokens(getattr(task, "name", "")) | _tokens(getattr(task, "description", ""))
        op_type = _op_type(task)
        backend = str(getattr(task, "backend", "") or "")
        ranked = []
        for index, entry in enumerate(self.entries):
            score = 0
            entry_backend = str(entry.get("backend") or "")
            scope = _memory_scope(entry)
            # Never transfer backend-specific implementation advice across
            # platforms. Generic/structural lessons remain eligible, but at a
            # lower priority than an exact backend match.
            if entry_backend and backend and entry_backend != backend and scope == "backend_specific":
                continue
            if entry.get("op_type") == op_type and op_type != "unknown":
                score += 8
            if entry_backend == backend and backend:
                score += 2
            elif scope == "generic":
                score += 1
            score += 3 * len(task_tokens & _tokens(entry.get("task_name")))
            score += len(task_tokens & _tokens(entry.get("tags")))
            if score:
                item = dict(entry)
                item["memory_scope"] = scope
                item["retrieval_score"] = score
                ranked.append((score, index, item))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _score, _index, entry in ranked[: self.top_k]]

    def record(self, task, result) -> None:
        if self.path is None:
            return
        candidates = [node for node_id, node in result.nodes.items() if node_id != "root"]
        selected = sorted([n for n in candidates if n.correct], key=lambda n: (n.speedup is None, -(n.speedup or 0)))[:2]
        selected += [n for n in candidates if not n.correct][:2]
        for node in selected:
            self.entries.append({
                "task_name": task.name,
                "op_type": _op_type(task),
                "backend": task.backend,
                "memory_scope": "backend_specific" if task.backend else "generic",
                "tags": list(getattr(task, "tags", []) or []),
                "strategy": node.plan_strategy_name,
                "outcome": "correct" if node.correct else "failed",
                "failure_stage": node.latest_failure_stage,
                "failure_type": node.failure_type,
                "speedup": node.speedup,
                "lesson": _lesson(node),
            })
        self.entries = self.entries[-256:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"version": 1, "entries": self.entries}, indent=2, ensure_ascii=False), encoding="utf-8")


# Compatibility import for experiment scripts and manifests created before the
# operator-memory/experience-journal split.
LongTermMemory = TaskExperienceJournal
