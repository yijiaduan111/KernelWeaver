#!/usr/bin/env python3
"""Audit/freeze public knowledge and optionally report read-only task coverage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.knowledge.public_wiki import PUBLIC_ROOT, PublicKernelWiki, audit_public_corpus, corpus_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--freeze", action="store_true", help="Explicitly freeze reviewed source pages and retrieval data")
    parser.add_argument("--manifest", type=Path, help="Optional task manifest; coverage only, no execution or generation")
    parser.add_argument("--report", type=Path, help="Write an audit artifact outside the corpus")
    args = parser.parse_args()
    if args.report and args.report.resolve().is_relative_to(args.wiki.resolve()):
        parser.error("Coverage artifacts must stay outside the public corpus")
    if args.freeze:
        audit_public_corpus(args.wiki, check_lock=False)
        manifest = corpus_manifest(args.wiki)
        manifest["frozen_at"] = datetime.now(timezone.utc).isoformat()
        (args.wiki / "corpus.lock.json").write_text(json.dumps(manifest, indent=2) + "\n")
    audit = audit_public_corpus(args.wiki)
    result = {"pages": audit["pages"], "knowledge_scope": audit["scope"], "corpus_sha256": audit["sha256"]}
    if args.manifest:
        from src.core.loader import KernelBenchLoader
        from src.experiment.batch_runner import load_task_manifest
        manifest = load_task_manifest(args.manifest, ROOT / "KernelBench", "cuda")
        wiki = PublicKernelWiki(args.wiki, target_arch="sm86")
        rows = []
        for entry in manifest["tasks"]:
            task = KernelBenchLoader().load_official_problem(ROOT / "KernelBench", entry["level"], entry["problem_id"], backend=entry["backend"])
            evidence = wiki.retrieve(task)
            pages = [p for p in evidence if p["page_type"] == "kernel"]
            detected = wiki.task_families(task)
            covered = {family for page in pages for family in page["operator_families"]}
            rows.append({"level": entry["level"], "problem_id": entry["problem_id"],
                         "matched_operator_pages": [p["id"] for p in pages],
                         "detected_families": sorted(detected),
                         "families_outside_context_budget": sorted(detected - covered),
                         "specialized_coverage": bool(pages),
                         "evidence_ids": [p["id"] for p in evidence]})
        result.update({"task_count": len(rows), "matched_task_count": sum(r["specialized_coverage"] for r in rows),
                       "tasks": rows, "note": "Retrieval coverage only; no correctness or speed claims. This is a development-set audit."})
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
