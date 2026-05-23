#!/usr/bin/env python3
"""Inspect KernelWeaver run artifacts.

This script is intentionally read-only. It summarizes semantic profiles,
strategy portfolios, strategy usage, and failure patterns from run.json files.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def short(value: Any, limit: int = 220) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def task_name(path: Path) -> str:
    return path.parent.name


def find_run_jsons(run_dir: Path, task_filter: str | None) -> list[Path]:
    if run_dir.is_file() and run_dir.name == "run.json":
        candidates = [run_dir]
    else:
        candidates = sorted(run_dir.glob("*/run.json"))
    if not task_filter:
        return candidates
    needle = task_filter.lower()
    return [path for path in candidates if needle in str(path).lower()]


def speedup(data: dict[str, Any]) -> Any:
    speedups = data.get("speedups")
    if isinstance(speedups, dict):
        if "torch_eager" in speedups:
            return speedups["torch_eager"]
        numeric = [value for value in speedups.values() if isinstance(value, (int, float))]
        return max(numeric) if numeric else None
    return data.get("best_speedup") or data.get("speedup")


def nodes(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = data.get("nodes") or {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if isinstance(value, dict)}
    if isinstance(raw, list):
        return {str(index): value for index, value in enumerate(raw) if isinstance(value, dict)}
    return {}


def format_heading(title: str, level: int = 2) -> list[str]:
    return ["#" * level + " " + title, ""]


def emit_summary(run_dir: Path, run_jsons: list[Path]) -> list[str]:
    lines: list[str] = []
    lines += format_heading("Run Summary", 1)
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Tasks with run.json: `{len(run_jsons)}`")
    summary = run_dir / "summary.json" if run_dir.is_dir() else None
    if summary and summary.exists():
        data = load_json(summary)
        aggregates = data.get("aggregates") or {}
        lines.append(f"- Summary rows: `{len(data.get('rows') or [])}`")
        for key in ["task_count", "success_count", "compile_rate", "correct_rate", "best_speedup", "median_speedup", "improved_over_reference_rate"]:
            if key in aggregates:
                lines.append(f"- {key}: `{aggregates[key]}`")
        paper = aggregates.get("paper_metrics")
        if paper:
            lines.append(f"- paper_metrics: `{short(paper, 500)}`")
    lines.append("")
    if run_jsons:
        lines.append("| Task | Speedup | Best | Attempts | Compile OK | Correct | Strategies | Op Type |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---|")
        for path in run_jsons:
            data = load_json(path)
            task_nodes = nodes(data)
            attempts = (data.get("stats") or {}).get("attempt_count")
            compile_ok = sum(1 for node in task_nodes.values() if node.get("compile_ok") is True)
            correct = sum(1 for node in task_nodes.values() if node.get("correct") is True)
            strategies = len(((data.get("strategy_portfolio") or {}).get("strategies")) or [])
            op_type = (data.get("semantic_profile") or {}).get("op_type")
            lines.append(
                f"| `{task_name(path)}` | `{speedup(data)}` | `{data.get('best_node_id')}` | `{attempts}` | `{compile_ok}` | `{correct}` | `{strategies}` | `{op_type}` |"
            )
        lines.append("")
    return lines


def emit_semantic(data: dict[str, Any]) -> list[str]:
    semantic = data.get("semantic_profile") or {}
    lines = format_heading("Semantic Profile", 2)
    if not semantic:
        return lines + ["No semantic_profile found.", ""]
    lines.append(f"- Enabled: `{semantic.get('enabled')}`")
    lines.append(f"- Mode: `{semantic.get('mode')}`")
    lines.append(f"- Op type: `{semantic.get('op_type')}`")
    lines.append(f"- Summary: {semantic.get('summary') or '-'}")
    lines.append(f"- Recommended anchors: `{short(semantic.get('recommended_anchors'), 600)}`")
    lines.append(f"- Risk notes: `{short(semantic.get('risk_notes'), 800)}`")
    intents = semantic.get("optimization_intents") or []
    if intents:
        lines.append("")
        lines.append("| # | Intent | Priority | Target Anchors | Summary |")
        lines.append("|---:|---|---:|---|---|")
        for index, intent in enumerate(intents, 1):
            lines.append(
                f"| {index} | `{intent.get('name')}` | `{intent.get('priority')}` | `{short(intent.get('target_anchors'), 300)}` | {short(intent.get('summary'), 220)} |"
            )
    anchors = semantic.get("anchors") or []
    if anchors:
        lines.append("")
        lines.append("Anchor hints:")
        for anchor in anchors[:12]:
            if isinstance(anchor, dict):
                lines.append(f"- `{anchor.get('anchor_name') or anchor.get('name')}`: {short(anchor, 500)}")
            else:
                lines.append(f"- {short(anchor, 500)}")
    lines.append("")
    return lines


def emit_strategy_portfolio(data: dict[str, Any]) -> list[str]:
    portfolio = data.get("strategy_portfolio") or {}
    lines = format_heading("Strategy Portfolio", 2)
    if not portfolio:
        return lines + ["No strategy_portfolio found.", ""]
    lines.append(f"- Enabled: `{portfolio.get('enabled')}`")
    lines.append(f"- Mode: `{portfolio.get('mode')}`")
    lines.append(f"- Providers: `{short(portfolio.get('providers'), 500)}`")
    lines.append(f"- Proposal errors: `{short(portfolio.get('proposal_errors'), 900)}`")
    lines.append(f"- Review errors: `{short(portfolio.get('review_errors'), 900)}`")
    strategies = portfolio.get("strategies") or []
    lines.append(f"- Strategy count: `{len(strategies)}`")
    if strategies:
        lines.append("")
        lines.append("| ID | Intent | Source | Scores | Anchors | Summary |")
        lines.append("|---|---|---|---|---|---|")
        for strategy in strategies:
            sid = strategy.get("strategy_id") or strategy.get("id")
            lines.append(
                f"| `{sid}` | `{short(strategy.get('intent'), 120)}` | `{short(strategy.get('source_models'), 180)}` | `{short(strategy.get('model_scores'), 220)}` | `{short(strategy.get('target_anchors'), 220)}` | {short(strategy.get('summary'), 320)} |"
            )
        lines.append("")
        lines.append("Implementation hints and risks:")
        for strategy in strategies:
            sid = strategy.get("strategy_id") or strategy.get("id")
            hints = strategy.get("implementation_hints") or strategy.get("backend_hint")
            risks = strategy.get("risk_notes") or strategy.get("risks")
            if hints or risks:
                lines.append(f"- `{sid}` hints: {short(hints, 700)}")
                lines.append(f"- `{sid}` risks: {short(risks, 500)}")
    lines.append("")
    return lines


def emit_strategy_usage(data: dict[str, Any]) -> list[str]:
    task_nodes = nodes(data)
    lines = format_heading("Strategy Usage By Attempt", 2)
    if not task_nodes:
        return lines + ["No nodes found.", ""]
    lines.append("| Node | Strategy | Correct | Compile | Speedup | Failure | Stage | Summary |")
    lines.append("|---|---|---:|---:|---:|---|---|---|")
    for node_id, node in task_nodes.items():
        if node_id == "root":
            continue
        lines.append(
            f"| `{node_id}` | `{node.get('plan_strategy_name')}` | `{node.get('correct')}` | `{node.get('compile_ok')}` | `{node.get('speedup')}` | `{node.get('failure_type')}` | `{node.get('latest_failure_stage')}` | {short(node.get('plan_summary'), 280)} |"
        )
    lines.append("")
    return lines


def emit_failure_breakdown(data: dict[str, Any]) -> list[str]:
    lines = format_heading("Failure Breakdown", 2)
    stats = data.get("stats") or {}
    lines.append(f"- Stats: `{short(stats, 1000)}`")
    task_nodes = nodes(data)
    failure_counts = Counter()
    stage_counts = Counter()
    log_snippets: list[tuple[str, str, str]] = []
    for node_id, node in task_nodes.items():
        failure = node.get("failure_type")
        stage = node.get("latest_failure_stage")
        if failure:
            failure_counts[str(failure)] += 1
        if stage:
            stage_counts[str(stage)] += 1
        logs = node.get("logs") or []
        if isinstance(logs, list):
            for log in logs[-2:]:
                if log:
                    log_snippets.append((node_id, str(failure), str(log)))
    lines.append(f"- Failure counts from nodes: `{dict(failure_counts)}`")
    lines.append(f"- Stage counts from nodes: `{dict(stage_counts)}`")
    if log_snippets:
        lines.append("")
        lines.append("Recent node log snippets:")
        for node_id, failure, log in log_snippets[:20]:
            lines.append(f"- `{node_id}` `{failure}`: {short(log, 500)}")
    lines.append("")
    return lines


def emit_code_locations(data: dict[str, Any], run_json: Path) -> list[str]:
    lines = format_heading("Code Artifacts", 2)
    best_code = run_json.parent / "best_code.py"
    lines.append(f"- run.json: `{run_json}`")
    lines.append(f"- best_code.py: `{best_code if best_code.exists() else 'missing'}`")
    best = data.get("best_node_id")
    if best:
        node = nodes(data).get(str(best))
        if node:
            lines.append(f"- best_node_id: `{best}`")
            lines.append(f"- best plan: {short(node.get('plan_summary'), 600)}")
            lines.append(f"- best strategy: `{node.get('plan_strategy_name')}`")
    lines.append("")
    return lines


def inspect_task(run_json: Path) -> list[str]:
    data = load_json(run_json)
    lines = format_heading(task_name(run_json), 1)
    lines.append(f"- Task name: `{data.get('task_name')}`")
    lines.append(f"- Level/problem: `{data.get('level')}/{data.get('problem_id')}`")
    lines.append(f"- Backend: `{data.get('backend')}`")
    lines.append(f"- Best node: `{data.get('best_node_id')}`")
    lines.append(f"- Speedup: `{speedup(data)}`")
    lines.append(f"- Source origin: `{data.get('source_origin')}`")
    lines.append("")
    lines += emit_semantic(data)
    lines += emit_strategy_portfolio(data)
    lines += emit_strategy_usage(data)
    lines += emit_failure_breakdown(data)
    lines += emit_code_locations(data, run_json)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KernelWeaver run artifacts.")
    parser.add_argument("run_dir", type=Path, help="Run directory or a task run.json path.")
    parser.add_argument("--task", help="Substring filter for task directory or run.json path.")
    parser.add_argument("--output", type=Path, help="Write markdown report to this path.")
    parser.add_argument("--summary-only", action="store_true", help="Only print the run-level summary table.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_jsons = find_run_jsons(run_dir, args.task)
    lines = emit_summary(run_dir, run_jsons)
    if not args.summary_only:
        for run_json in run_jsons:
            lines += inspect_task(run_json)
    report = "\n".join(lines).rstrip() + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
