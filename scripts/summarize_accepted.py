from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_ROOT = REPO_ROOT / "runs_final" / "kernelbench_k10"
PRIMARY_REFERENCE = "torch_eager"
EXPECTED_COUNTS = {1: 100, 2: 100, 3: 50}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize accepted KernelBench K=10 results.")
    parser.add_argument("--final-root", default=str(DEFAULT_FINAL_ROOT), help="Accepted result root.")
    parser.add_argument("--k", type=int, default=10, help="Attempt budget used in the accepted run set.")
    parser.add_argument("--mode", default=PRIMARY_REFERENCE, help="Reference mode to summarize.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _fmt_level(level: int) -> str:
    return f"level{level}"


def _speedup(node: dict[str, Any], mode: str) -> float | None:
    speedups = node.get("speedups")
    value = speedups.get(mode) if isinstance(speedups, dict) else None
    if value is None and mode == PRIMARY_REFERENCE:
        value = node.get("speedup")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _attempt_sort_key(node_id: str, ordinal: int) -> tuple[int, int, int]:
    if node_id == "root":
        return (0, 0, ordinal)
    match = re.fullmatch(r"n(\d+)", node_id)
    if match:
        return (1, int(match.group(1)), ordinal)
    match = re.fullmatch(r"phase2_n(\d+)", node_id)
    if match:
        return (2, int(match.group(1)), ordinal)
    numbers = re.findall(r"\d+", node_id)
    if numbers:
        return (3, int(numbers[-1]), ordinal)
    return (4, ordinal, ordinal)


def _is_real_candidate(node_id: str, node: dict[str, Any]) -> bool:
    if node_id == "root":
        return False
    origin = str(node.get("origin") or "")
    return origin not in {"root", "phase2_root"}


def _generated_nodes(payload: dict[str, Any], k: int | None = None) -> list[tuple[str, dict[str, Any]]]:
    nodes = payload.get("nodes")
    if not isinstance(nodes, dict):
        return []
    generated = [
        (node_id, node)
        for node_id, node in nodes.items()
        if isinstance(node, dict) and _is_real_candidate(node_id, node)
    ]
    ordered = sorted(generated, key=lambda item: _attempt_sort_key(item[0], list(nodes).index(item[0])))
    return ordered[:k] if k is not None else ordered


def _best_correct(generated: list[tuple[str, dict[str, Any]]], mode: str) -> tuple[str, dict[str, Any], float] | None:
    correct = []
    for node_id, node in generated:
        if bool(node.get("correct")):
            speedup = _speedup(node, mode)
            if speedup is not None:
                correct.append((node_id, node, speedup))
    if not correct:
        return None
    return max(correct, key=lambda item: item[2])


def _first_correct(generated: list[tuple[str, dict[str, Any]]], mode: str) -> tuple[int, str, float] | None:
    for index, (node_id, node) in enumerate(generated, start=1):
        if bool(node.get("correct")):
            speedup = _speedup(node, mode)
            return (index, node_id, speedup if speedup is not None else 0.0)
    return None


def _run_row(run_dir: Path, mode: str, k: int) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    payload = _read_json(run_path)
    level = int(payload["level"])
    problem_id = int(payload["problem_id"])
    generated = _generated_nodes(payload, k=k)
    compile_count = sum(1 for _node_id, node in generated if bool(node.get("compile_ok")))
    correct_count = sum(1 for _node_id, node in generated if bool(node.get("correct")))
    first = _first_correct(generated, mode)
    best = _best_correct(generated, mode)
    best_speedup = best[2] if best is not None else None
    first_speedup = first[2] if first is not None else None
    return {
        "level": level,
        "problem_id": problem_id,
        "task_name": payload.get("task_name"),
        "run_dir": str(run_dir),
        "candidate_count": len(generated),
        "compile_count": compile_count,
        "correct_count": correct_count,
        "success": best is not None,
        "fast1": bool(best_speedup is not None and best_speedup >= 1.0),
        "best_speedup": best_speedup,
        "best_node_id": best[0] if best is not None else None,
        "first_correct_attempt": first[0] if first is not None else None,
        "first_correct_node_id": first[1] if first is not None else None,
        "first_correct_speedup": first_speedup,
        "refine_gain": (best_speedup / first_speedup) if best_speedup is not None and first_speedup and first_speedup > 0 else None,
    }


def _collect_rows(final_root: Path, mode: str, k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in (1, 2, 3):
        for run_path in sorted((final_root / _fmt_level(level)).glob("L*_P*/run.json")):
            rows.append(_run_row(run_path.parent, mode, k))
    return sorted(rows, key=lambda row: (int(row["level"]), int(row["problem_id"])))


def _summarize(rows: list[dict[str, Any]], accepted_denominator: int, expected_total: int) -> dict[str, Any]:
    candidate_count = sum(int(row["candidate_count"]) for row in rows)
    compile_candidate_count = sum(int(row["compile_count"]) for row in rows)
    correct_candidate_count = sum(int(row["correct_count"]) for row in rows)
    compile_task_count = sum(1 for row in rows if int(row["compile_count"]) > 0)
    correct_task_count = sum(1 for row in rows if int(row["correct_count"]) > 0)
    solved = [row for row in rows if row["success"] and isinstance(row.get("best_speedup"), (int, float))]
    speedups = [float(row["best_speedup"]) for row in solved]
    first_correct = [int(row["first_correct_attempt"]) for row in solved if row.get("first_correct_attempt") is not None]
    refine_gains = [float(row["refine_gain"]) for row in solved if isinstance(row.get("refine_gain"), (int, float))]
    return {
        "accepted_tasks": accepted_denominator,
        "expected_tasks": expected_total,
        "candidate_count": candidate_count,
        "compile_task_count": compile_task_count,
        "correct_task_count": correct_task_count,
        "compile_rate": _ratio(compile_task_count, accepted_denominator),
        "correct_rate": _ratio(correct_task_count, accepted_denominator),
        "candidate_compile_rate": _ratio(compile_candidate_count, candidate_count),
        "candidate_correct_rate": _ratio(correct_candidate_count, candidate_count),
        "success_at_10": _ratio(sum(1 for row in rows if row["success"]), accepted_denominator),
        "fast1_at_10": _ratio(sum(1 for row in rows if row["fast1"]), accepted_denominator),
        "geomean_speedup_solved": _geomean(speedups),
        "median_speedup_solved": statistics.median(speedups) if speedups else None,
        "best_speedup": max(speedups) if speedups else None,
        "mean_speedup_solved": statistics.mean(speedups) if speedups else None,
        "first_correct_median": statistics.median(first_correct) if first_correct else None,
        "refine_gain_median": statistics.median(refine_gains) if refine_gains else None,
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "level",
        "problem_id",
        "task_name",
        "candidate_count",
        "compile_count",
        "correct_count",
        "success",
        "fast1",
        "best_speedup",
        "best_node_id",
        "first_correct_attempt",
        "first_correct_node_id",
        "first_correct_speedup",
        "refine_gain",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_summary_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket",
        "accepted_tasks",
        "expected_tasks",
        "candidate_count",
        "compile_task_count",
        "correct_task_count",
        "compile_rate",
        "correct_rate",
        "candidate_compile_rate",
        "candidate_correct_rate",
        "success_at_10",
        "fast1_at_10",
        "geomean_speedup_solved",
        "median_speedup_solved",
        "best_speedup",
        "mean_speedup_solved",
        "first_correct_median",
        "refine_gain_median",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bucket, summary in summaries.items():
            writer.writerow({"bucket": bucket, **{key: summary.get(key) for key in fieldnames if key != "bucket"}})


def _write_index(final_root: Path, rows: list[dict[str, Any]]) -> None:
    index_dir = final_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = index_dir / "accepted.tsv"
    missing_path = index_dir / "missing.tsv"
    with accepted_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("level\tproblem_id\trun_dir\tbest_speedup\tsuccess\tfast1\n")
        for row in rows:
            handle.write(
                f"{row['level']}\t{row['problem_id']}\t{row['run_dir']}\t{row.get('best_speedup') or ''}\t{row['success']}\t{row['fast1']}\n"
            )
    accepted_pairs = {(int(row["level"]), int(row["problem_id"])) for row in rows}
    with missing_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("level\tproblem_id\n")
        for level, expected in EXPECTED_COUNTS.items():
            for problem_id in range(1, expected + 1):
                if (level, problem_id) not in accepted_pairs:
                    handle.write(f"{level}\t{problem_id}\n")


def main() -> int:
    args = _parse_args()
    final_root = Path(args.final_root).resolve()
    rows = _collect_rows(final_root, args.mode, args.k)
    by_level: dict[str, dict[str, Any]] = {}
    for level in (1, 2, 3):
        level_rows = [row for row in rows if int(row["level"]) == level]
        by_level[f"level{level}"] = _summarize(level_rows, len(level_rows), EXPECTED_COUNTS[level])
    overall = _summarize(rows, len(rows), sum(EXPECTED_COUNTS.values()))
    reports_dir = final_root / "reports"
    payload = {"mode": args.mode, "k": args.k, "overall": overall, "by_level": by_level, "rows": rows}
    _write_json(reports_dir / "summary_overall.json", payload)
    _write_json(reports_dir / "summary_by_level.json", {"mode": args.mode, "k": args.k, "by_level": by_level})
    _write_summary_csv(reports_dir / "summary_overall.csv", {"overall": overall})
    _write_summary_csv(reports_dir / "summary_by_level.csv", by_level)
    _write_rows_csv(reports_dir / "accepted_rows.csv", rows)
    _write_index(final_root, rows)
    print(f"accepted_tasks={len(rows)}")
    print(f"summary={reports_dir / 'summary_overall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
