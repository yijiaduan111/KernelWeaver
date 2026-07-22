from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_ROOT = REPO_ROOT / "runs_final" / "kernelbench_k10"
PRIMARY_REFERENCE = "torch_eager"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accept one completed KernelBench run into runs_final/kernelbench_k10."
    )
    parser.add_argument("source_dir", help="Original single-task run directory containing run.json.")
    parser.add_argument("--final-root", default=str(DEFAULT_FINAL_ROOT), help="Final accepted-result root.")
    parser.add_argument("--k", type=int, default=10, help="Expected attempt budget.")
    parser.add_argument("--replace", action="store_true", help="Replace an existing accepted result.")
    parser.add_argument("--notes", default="manual reviewed", help="Notes written to source_attempt.txt.")
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow archiving a run with fewer generated candidates than K.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _generated_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = payload.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("run.json is missing a valid nodes object")
    generated = []
    for node_id, node in nodes.items():
        if node_id == "root" or not isinstance(node, dict):
            continue
        origin = str(node.get("origin") or "")
        if origin in {"root", "phase2_root"}:
            continue
        generated.append(node)
    return generated


def _speedup(node: dict[str, Any], mode: str = PRIMARY_REFERENCE) -> float | None:
    speedups = node.get("speedups")
    value = speedups.get(mode) if isinstance(speedups, dict) else None
    if value is None:
        value = node.get("speedup")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _best_correct_node(payload: dict[str, Any]) -> dict[str, Any] | None:
    correct_nodes = [node for node in _generated_nodes(payload) if bool(node.get("correct"))]
    if not correct_nodes:
        return None
    return max(correct_nodes, key=lambda node: _speedup(node) or 0.0)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _target_dir(final_root: Path, level: int, problem_id: int) -> Path:
    return final_root / f"level{level}" / f"L{level}_P{problem_id:03d}"


def _validate_run(payload: dict[str, Any], source_dir: Path, k: int, allow_short: bool) -> dict[str, Any]:
    level = payload.get("level")
    problem_id = payload.get("problem_id")
    if level is None or problem_id is None:
        raise ValueError("run.json must contain level and problem_id")
    level = int(level)
    problem_id = int(problem_id)
    if level not in {1, 2, 3}:
        raise ValueError(f"unsupported KernelBench level: {level}")

    generated = _generated_nodes(payload)
    if not generated:
        raise ValueError("run.json has no generated candidates")
    if len(generated) < k and not allow_short:
        raise ValueError(f"run has only {len(generated)} generated candidates; expected K={k}")

    best_correct = _best_correct_node(payload)
    best_speedup = _speedup(best_correct) if best_correct is not None else None
    return {
        "status": "accepted",
        "level": level,
        "problem_id": problem_id,
        "k": k,
        "source_dir": str(source_dir.resolve()),
        "accepted_reason": "completed_k10_valid" if len(generated) >= k else "completed_short_valid",
        "commit": _git_commit(),
        "run_profile": payload.get("run_profile") or (payload.get("config") or {}).get("run_profile"),
        "backend": payload.get("backend"),
        "code_provider": (payload.get("config") or {}).get("code_provider"),
        "generated_candidate_count": len(generated),
        "has_correct_candidate": best_correct is not None,
        "best_speedup": best_speedup,
        "accepted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _copy_optional(source_dir: Path, target_dir: Path, name: str) -> None:
    source = source_dir / name
    if source.exists() and source.is_file():
        shutil.copy2(source, target_dir / name)


def _materialize_best_code(source_dir: Path, target_dir: Path, payload: dict[str, Any]) -> None:
    best_code = source_dir / "best_code.py"
    if best_code.exists():
        shutil.copy2(best_code, target_dir / "best_code.py")
        return
    best_node_id = payload.get("best_node_id")
    node = (payload.get("nodes") or {}).get(best_node_id)
    if isinstance(node, dict) and isinstance(node.get("code"), str):
        (target_dir / "best_code.py").write_text(node["code"], encoding="utf-8")
        return
    raise ValueError("could not find best_code.py or recover best code from run.json")


def main() -> int:
    args = _parse_args()
    source_dir = Path(args.source_dir).resolve()
    run_path = source_dir / "run.json"
    if not run_path.exists():
        raise SystemExit(f"missing run.json: {run_path}")

    payload = _read_json(run_path)
    status = _validate_run(payload, source_dir, args.k, args.allow_short)
    final_root = Path(args.final_root).resolve()
    target = _target_dir(final_root, int(status["level"]), int(status["problem_id"]))
    if target.exists():
        if not args.replace:
            raise SystemExit(f"accepted result already exists: {target} (use --replace to overwrite)")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    shutil.copy2(run_path, target / "run.json")
    _materialize_best_code(source_dir, target, payload)
    _copy_optional(source_dir, target, "launcher.log")
    _copy_optional(source_dir, target, "driver.log")
    _write_json(target / "task_status.json", status)
    (target / "source_attempt.txt").write_text(
        "\n".join(
            [
                f"source_dir={status['source_dir']}",
                f"accepted_reason={status['accepted_reason']}",
                f"accepted_at={status['accepted_at']}",
                f"commit={status.get('commit') or ''}",
                f"notes={args.notes}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"accepted={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
