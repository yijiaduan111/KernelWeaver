"""Experiment and reporting helpers for saved STARK batch runs.

This module intentionally keeps experiment aggregation outside the search
loop so we can evolve paper-style tables and engineering summaries
without mutating the runtime workflow itself.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

PRIMARY_REFERENCE = "torch_eager"
REFERENCE_MODES = (
    PRIMARY_REFERENCE,
    "torch_compile_default",
    "torch_compile_max_autotune",
)
LEVEL_BUCKETS: tuple[tuple[str, int | None], ...] = (
    ("overall", None),
    ("L1", 1),
    ("L2", 2),
    ("L3", 3),
)
ABLATION_WORKFLOWS = ("sampling", "search-agent", "ma-only", "stark")


def load_batch_summary(path: str | Path) -> dict[str, Any]:
    """Load one saved batch `summary.json` payload."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_experiment_report(
    mock_summary: dict[str, Any],
    real_summary: dict[str, Any],
    rerun_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the engineering-facing experiment package.

    This keeps the previously shipped report contract intact while
    piggybacking on the richer paper-metric helpers introduced later.
    """
    manifest = real_summary.get("manifest") or mock_summary.get("manifest") or {"tasks": []}
    mock_rows = _row_index(mock_summary.get("rows", []))
    real_rows = _row_index(real_summary.get("rows", []))
    rerun_rows = _row_index((rerun_summary or {}).get("rows", []))
    task_rows: list[dict[str, Any]] = []

    for item in manifest.get("tasks", []):
        alias = str(item["alias"])
        real_row = real_rows.get(alias)
        rerun_row = rerun_rows.get(alias)
        final_row, final_source = _select_final_row(real_row, rerun_row)
        task_rows.append(
            {
                "alias": alias,
                "level": item["level"],
                "problem_id": item["problem_id"],
                "mock_speedup": _row_mode_speedup(mock_rows.get(alias), PRIMARY_REFERENCE),
                "mock_best_node": _row_value(mock_rows.get(alias), "best_node_id"),
                "mock_success": _row_best_correct(mock_rows.get(alias)),
                "mock_paper_fast1": _row_mode_fast1(mock_rows.get(alias), PRIMARY_REFERENCE),
                "real_speedup": _row_mode_speedup(real_row, PRIMARY_REFERENCE),
                "real_best_node": _row_value(real_row, "best_node_id"),
                "real_best_node_is_root": _row_value(real_row, "best_node_is_root"),
                "real_success": _row_best_correct(real_row),
                "real_paper_fast1": _row_mode_fast1(real_row, PRIMARY_REFERENCE),
                "rerun_speedup": _row_mode_speedup(rerun_row, PRIMARY_REFERENCE),
                "rerun_best_node": _row_value(rerun_row, "best_node_id"),
                "rerun_best_node_is_root": _row_value(rerun_row, "best_node_is_root"),
                "rerun_success": _row_best_correct(rerun_row),
                "rerun_paper_fast1": _row_mode_fast1(rerun_row, PRIMARY_REFERENCE),
                "final_source": final_source,
                "final_speedup": _row_mode_speedup(final_row, PRIMARY_REFERENCE),
                "final_best_node": _row_value(final_row, "best_node_id"),
                "final_best_node_is_root": _row_value(final_row, "best_node_is_root"),
                "final_improved_over_reference": _row_value(final_row, "improved_over_reference"),
                "final_status": _row_value(final_row, "status"),
                "final_success": _row_best_correct(final_row),
                "final_paper_fast1": _row_mode_fast1(final_row, PRIMARY_REFERENCE),
                "final_validation_correctness_matches": _row_value(final_row, "validation_correctness_matches"),
                "final_validation_speed_direction_matches": _row_value(final_row, "validation_speed_direction_matches"),
            }
        )

    final_rows = [row for row in task_rows if row["final_status"] == "ok"]
    return {
        "manifest_name": manifest.get("name", "unknown"),
        "task_count": len(manifest.get("tasks", [])),
        "cohorts": {
            "mock_max_attempts_2": _summary_metrics(mock_summary),
            "real_max_attempts_1": _summary_metrics(real_summary),
            "real_focused_rerun_max_attempts_3": _summary_metrics(rerun_summary) if rerun_summary else None,
            "final_best_after_rerun": _aggregate_final_rows(final_rows, len(manifest.get("tasks", []))),
        },
        "task_rows": task_rows,
    }


def build_paper_summary_report(named_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a paper-style summary package with Table 1/2/3 style views."""
    cohorts: dict[str, Any] = {}
    table1_rows: list[dict[str, Any]] = []
    table2_rows: list[dict[str, Any]] = []

    for label, summary in named_summaries.items():
        overall = {mode: build_paper_metrics(summary, mode=mode) for mode in REFERENCE_MODES}
        by_level = {
            bucket_name: {mode: build_paper_metrics(summary, mode=mode, level=level) for mode in REFERENCE_MODES}
            for bucket_name, level in LEVEL_BUCKETS
        }
        compile_correct_overall = build_compile_correct_metrics(summary)
        compile_correct_by_level = {
            bucket_name: build_compile_correct_metrics(summary, level=level)
            for bucket_name, level in LEVEL_BUCKETS
        }
        cohort = {
            "paper_metrics": overall[PRIMARY_REFERENCE],
            "paper_metrics_by_mode": overall,
            "paper_metrics_by_level": by_level,
            "compile_correct_metrics": compile_correct_overall,
            "compile_correct_metrics_by_level": compile_correct_by_level,
            "engineering_metrics": _summary_metrics(summary),
            "workflow": summary.get("workflow", "stark"),
            "provider": summary.get("provider"),
            "manifest_name": (summary.get("manifest") or {}).get("name"),
        }
        cohorts[label] = cohort
        workflow = cohort["workflow"]
        provider = cohort.get("provider") or "n/a"
        manifest_name = cohort.get("manifest_name") or "n/a"
        for bucket_name, _level in LEVEL_BUCKETS:
            for mode in REFERENCE_MODES:
                metrics = by_level[bucket_name][mode]
                table1_rows.append(
                    {
                        "label": label,
                        "workflow": workflow,
                        "provider": provider,
                        "manifest": manifest_name,
                        "level_bucket": bucket_name,
                        "reference_mode": mode,
                        **(metrics or {"task_count": 0, "Success": None, "Fast1": None, "Speed": None, "median_speedup": None, "best_speedup": None}),
                    }
                )
            compile_metrics = compile_correct_by_level[bucket_name]
            table2_rows.append(
                {
                    "label": label,
                    "workflow": workflow,
                    "provider": provider,
                    "manifest": manifest_name,
                    "level_bucket": bucket_name,
                    **(compile_metrics or {"candidate_count": 0, "compile_rate": None, "correct_rate": None}),
                }
            )

    table3_rows = _build_ablation_rows(cohorts)
    return {
        "cohorts": cohorts,
        "labels": list(named_summaries.keys()),
        "table1": table1_rows,
        "table2": table2_rows,
        "table3": table3_rows,
    }


def write_experiment_report(report: dict[str, Any], output_dir: str | Path, title: str = "KernelBench Small-Scale Experiment") -> dict[str, Path]:
    """Write the engineering-facing experiment report as JSON and Markdown."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "experiment_report.json"
    md_path = root / "experiment_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown_report(report, title), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def write_paper_summary_report(
    report: dict[str, Any],
    output_dir: str | Path,
    title: str = "STARK Paper-Aligned Summary",
) -> dict[str, Path]:
    """Write the paper-style summary report as JSON and Markdown."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "paper_summary_report.json"
    md_path = root / "paper_summary_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_paper_markdown_report(report, title), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def build_paper_metrics(
    summary: dict[str, Any] | None,
    mode: str = PRIMARY_REFERENCE,
    level: int | None = None,
) -> dict[str, Any] | None:
    """Compute `Success`, `Fast1`, and `Speed` for one summary slice."""
    rows = _filtered_rows(summary, level=level)
    total = len(rows)
    if total == 0:
        return {
            "task_count": 0,
            "Success": None,
            "Fast1": None,
            "Speed": None,
            "median_speedup": None,
            "best_speedup": None,
            "failure_stage_distribution": {},
        }
    successful = [row for row in rows if _row_best_correct(row)]
    speedups = [
        float(_row_mode_speedup(row, mode))
        for row in successful
        if isinstance(_row_mode_speedup(row, mode), (int, float))
    ]
    speed_metric_values = [
        float(_row_mode_speedup(row, mode))
        if _row_best_correct(row) and isinstance(_row_mode_speedup(row, mode), (int, float))
        else 0.0
        for row in rows
    ]
    failure_stage_distribution: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("failure_stage") or "unknown")
        failure_stage_distribution[stage] = failure_stage_distribution.get(stage, 0) + 1
    return {
        "task_count": total,
        "Success": _ratio(len(successful), total),
        "Fast1": _ratio(sum(1 for row in rows if _row_mode_fast1(row, mode)), total),
        "Speed": (sum(speed_metric_values) / total) if total > 0 else None,
        "median_speedup": statistics.median(speedups) if speedups else None,
        "best_speedup": max(speedups) if speedups else None,
        "failure_stage_distribution": failure_stage_distribution,
    }


def build_compile_correct_metrics(summary: dict[str, Any] | None, level: int | None = None) -> dict[str, Any] | None:
    """Compute Table 2 style compile/correct rates over non-root candidates."""
    rows = _filtered_rows(summary, level=level)
    if not rows:
        return {
            "candidate_count": 0,
            "compile_rate": None,
            "correct_rate": None,
        }
    total = sum(int(row.get("candidate_total_count") or 0) for row in rows)
    compile_count = sum(int(row.get("candidate_compile_count") or 0) for row in rows)
    correct_count = sum(int(row.get("candidate_correct_count") or 0) for row in rows)
    return {
        "candidate_count": total,
        "compile_rate": _ratio(compile_count, total),
        "correct_rate": _ratio(correct_count, total),
    }


def _summary_metrics(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    aggregates = dict(summary.get("aggregates") or {})
    rows = list(summary.get("rows", []))
    total = len(rows)
    if "best_node_is_root_rate" not in aggregates:
        successful = [row for row in rows if row.get("status") == "ok"]
        aggregates["best_node_is_root_rate"] = _ratio(sum(1 for row in successful if row.get("best_node_id") == "root"), total)
    if "paper_metrics" not in aggregates:
        aggregates["paper_metrics"] = build_paper_metrics(summary, mode=PRIMARY_REFERENCE)
    if "paper_metrics_by_mode" not in aggregates:
        aggregates["paper_metrics_by_mode"] = {mode: build_paper_metrics(summary, mode=mode) for mode in REFERENCE_MODES}
    if "compile_rate" not in aggregates or "correct_rate" not in aggregates:
        aggregates.update(build_compile_correct_metrics(summary) or {})
    return aggregates


def _aggregate_final_rows(rows: list[dict[str, Any]], total_tasks: int) -> dict[str, Any]:
    speedups = [float(row["final_speedup"]) for row in rows if isinstance(row.get("final_speedup"), (int, float))]
    failure_stage_distribution: dict[str, int] = {}
    for _row in rows:
        stage = "none"
        failure_stage_distribution[stage] = failure_stage_distribution.get(stage, 0) + 1
    speed_metric_values = [
        float(row["final_speedup"]) if row.get("final_success") and isinstance(row.get("final_speedup"), (int, float)) else 0.0
        for row in rows
    ]
    return {
        "task_count": total_tasks,
        "success_count": sum(1 for row in rows if row.get("final_success")),
        "root_correct_rate": None,
        "non_root_correct_candidate_rate": _ratio(sum(1 for row in rows if not row.get("final_best_node_is_root")), total_tasks),
        "improved_over_reference_rate": _ratio(sum(1 for row in rows if row.get("final_improved_over_reference")), total_tasks),
        "best_node_is_root_rate": _ratio(sum(1 for row in rows if row.get("final_best_node_is_root")), total_tasks),
        "median_speedup": statistics.median(speedups) if speedups else None,
        "best_speedup": max(speedups) if speedups else None,
        "failure_stage_distribution": failure_stage_distribution,
        "paper_metrics": {
            "Success": _ratio(sum(1 for row in rows if row.get("final_success")), total_tasks),
            "Fast1": _ratio(sum(1 for row in rows if row.get("final_paper_fast1")), total_tasks),
            "Speed": (sum(speed_metric_values) / total_tasks) if total_tasks > 0 else None,
        },
    }


def _render_markdown_report(report: dict[str, Any], title: str) -> str:
    cohorts = report["cohorts"]
    lines = [f"# {title}", "", "## Cohort Metrics", ""]
    lines.append(
        "| Cohort | task_count | success_count | Success | Fast1 | Speed | non_root_correct_candidate_rate | improved_over_reference_rate | median_speedup | best_speedup | best_node_is_root_rate |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, payload in cohorts.items():
        if payload is None:
            continue
        paper_metrics = payload.get("paper_metrics") or {}
        lines.append(
            "| {label} | {task_count} | {success_count} | {success} | {fast1} | {speed} | {non_root} | {improved} | {median} | {best} | {root_rate} |".format(
                label=label,
                task_count=payload.get("task_count", "n/a"),
                success_count=payload.get("success_count", "n/a"),
                success=_fmt_ratio(paper_metrics.get("Success")),
                fast1=_fmt_ratio(paper_metrics.get("Fast1")),
                speed=_fmt_speedup(paper_metrics.get("Speed")),
                non_root=_fmt_ratio(payload.get("non_root_correct_candidate_rate")),
                improved=_fmt_ratio(payload.get("improved_over_reference_rate")),
                median=_fmt_speedup(payload.get("median_speedup")),
                best=_fmt_speedup(payload.get("best_speedup")),
                root_rate=_fmt_ratio(payload.get("best_node_is_root_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Task Table",
            "",
            "| Alias | Real v1 | Rerun v3 | Final | Final source | Final best node | Final Success | Final Fast1 | Final improved |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in report["task_rows"]:
        lines.append(
            "| {alias} | {real} | {rerun} | {final} | {source} | {best_node} | {success} | {fast1} | {improved} |".format(
                alias=row["alias"],
                real=_fmt_speedup(row.get("real_speedup")),
                rerun=_fmt_speedup(row.get("rerun_speedup")),
                final=_fmt_speedup(row.get("final_speedup")),
                source=row.get("final_source") or "n/a",
                best_node=row.get("final_best_node") or "n/a",
                success="yes" if row.get("final_success") else "no",
                fast1="yes" if row.get("final_paper_fast1") else "no",
                improved="yes" if row.get("final_improved_over_reference") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This package is a small-scale credible experiment set, not a paper-level reproduction.",
            "- `Success`, `Fast1`, and `Speed` are paper-style metrics layered on top of the engineering summary.",
            "- `final` chooses the faster correct result between the real v1 run and the focused rerun when both exist.",
            "- `non_root_correct_candidate_rate` and `best_node_is_root_rate` help show whether search is actually beating the root candidate.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_paper_markdown_report(report: dict[str, Any], title: str) -> str:
    lines = [f"# {title}", "", "## Table 1 Style Metrics", ""]
    lines.append("| Label | Workflow | Level | Reference | task_count | Success | Fast1 | Speed | median_speedup | best_speedup |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["table1"]:
        lines.append(
            "| {label} | {workflow} | {level_bucket} | {reference_mode} | {task_count} | {success} | {fast1} | {speed} | {median} | {best} |".format(
                label=row["label"],
                workflow=row["workflow"],
                level_bucket=row["level_bucket"],
                reference_mode=row["reference_mode"],
                task_count=row.get("task_count", 0),
                success=_fmt_ratio(row.get("Success")),
                fast1=_fmt_ratio(row.get("Fast1")),
                speed=_fmt_speedup(row.get("Speed")),
                median=_fmt_speedup(row.get("median_speedup")),
                best=_fmt_speedup(row.get("best_speedup")),
            )
        )

    lines.extend(["", "## Table 2 Style Metrics", ""])
    lines.append("| Label | Workflow | Level | candidate_count | Compile Rate | Correct Rate |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for row in report["table2"]:
        lines.append(
            "| {label} | {workflow} | {level_bucket} | {candidate_count} | {compile_rate} | {correct_rate} |".format(
                label=row["label"],
                workflow=row["workflow"],
                level_bucket=row["level_bucket"],
                candidate_count=row.get("candidate_count", 0),
                compile_rate=_fmt_ratio(row.get("compile_rate")),
                correct_rate=_fmt_ratio(row.get("correct_rate")),
            )
        )

    lines.extend(["", "## Table 3 Style Ablation", ""])
    lines.append("| Workflow | task_count | Success | Fast1 | Speed |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report["table3"]:
        lines.append(
            "| {workflow} | {task_count} | {success} | {fast1} | {speed} |".format(
                workflow=row["workflow"],
                task_count=row.get("task_count", 0),
                success=_fmt_ratio(row.get("Success")),
                fast1=_fmt_ratio(row.get("Fast1")),
                speed=_fmt_speedup(row.get("Speed")),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Success`: final best candidate compiles and is correct.",
            "- `Fast1`: final best candidate is correct and has `speedup >= 1.0` for the selected reference mode.",
            "- `Speed`: average final speedup with failures counted as `0.0`.",
            "- Table 2 uses all non-root generated candidates, not just the final best node.",
            "- This report is paper-aligned engineering output, not a claim of full paper reproduction.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_ablation_rows(cohorts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workflow in ABLATION_WORKFLOWS:
        matching = [payload for payload in cohorts.values() if payload.get("workflow") == workflow]
        if not matching:
            continue
        metrics = matching[0]["paper_metrics_by_level"]["overall"][PRIMARY_REFERENCE]
        rows.append(
            {
                "workflow": workflow,
                "task_count": metrics.get("task_count", 0),
                "Success": metrics.get("Success"),
                "Fast1": metrics.get("Fast1"),
                "Speed": metrics.get("Speed"),
            }
        )
    return rows


def _row_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["alias"]): row for row in rows}


def _filtered_rows(summary: dict[str, Any] | None, level: int | None = None) -> list[dict[str, Any]]:
    if not summary:
        return []
    rows = list(summary.get("rows", []))
    if level is None:
        return rows
    return [row for row in rows if int(row.get("level") or 0) == level]


def _select_final_row(real_row: dict[str, Any] | None, rerun_row: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if rerun_row and rerun_row.get("status") == "ok":
        if not real_row or real_row.get("status") != "ok":
            return rerun_row, "rerun"
        real_speedup = _numeric(_row_mode_speedup(real_row, PRIMARY_REFERENCE))
        rerun_speedup = _numeric(_row_mode_speedup(rerun_row, PRIMARY_REFERENCE))
        if rerun_speedup is not None and real_speedup is not None and rerun_speedup >= real_speedup:
            return rerun_row, "rerun"
    if real_row and real_row.get("status") == "ok":
        return real_row, "real"
    if rerun_row and rerun_row.get("status") == "ok":
        return rerun_row, "rerun"
    return real_row or rerun_row, "none"


def _row_best_correct(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    return bool(row.get("status") == "ok" and row.get("best_correct"))


def _row_mode_speedup(row: dict[str, Any] | None, mode: str) -> float | None:
    if row is None:
        return None
    key = f"{mode}_speedup"
    if key in row and isinstance(row.get(key), (int, float)):
        return float(row[key])
    if mode == PRIMARY_REFERENCE and isinstance(row.get("speedup"), (int, float)):
        return float(row["speedup"])
    return None


def _row_mode_fast1(row: dict[str, Any] | None, mode: str) -> bool:
    speedup = _row_mode_speedup(row, mode)
    return bool(_row_best_correct(row) and isinstance(speedup, (int, float)) and speedup >= 1.0)


def _row_value(row: dict[str, Any] | None, key: str) -> Any:
    if row is None:
        return None
    return row.get(key)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}x"
