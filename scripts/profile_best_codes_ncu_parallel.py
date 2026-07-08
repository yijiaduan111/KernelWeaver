from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SINGLE_TASK_SCRIPT = REPO_ROOT / "scripts" / "profile_best_codes_ncu.py"


def _load_single_task_module():
    spec = importlib.util.spec_from_file_location("profile_best_codes_ncu", SINGLE_TASK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SINGLE_TASK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_single_task_module = _load_single_task_module()
BEST_CODE_MANIFEST: list[dict[str, Any]] = list(_single_task_module.BEST_CODE_MANIFEST)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run best-code NCU profiling in isolated parallel workers.")
    parser.add_argument("--kernelbench-root", default="KernelBench")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids.")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--profile-runs", type=int, default=3)
    parser.add_argument("--only", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"ncu_best_codes_parallel_{stamp}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _append_summary(path: Path, row: dict[str, Any]) -> None:
    header = [
        "problem_id",
        "label",
        "gpu",
        "status",
        "best_speedup",
        "kernel_name",
        "row_count",
        "best_code",
        "profile_dir",
        "log_path",
        "returncode",
        "error",
    ]
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write("\t".join(header) + "\n")
        handle.write("\t".join(str(row.get(key, "") or "") for key in header) + "\n")


def _selected_entries(only: str) -> list[dict[str, Any]]:
    if not only.strip():
        return list(BEST_CODE_MANIFEST)
    wanted = {int(item.strip().lstrip("Pp")) for item in only.split(",") if item.strip()}
    return [entry for entry in BEST_CODE_MANIFEST if int(entry["problem_id"]) in wanted]


def _parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required.")
    return gpus


def _task_name(entry: dict[str, Any]) -> str:
    return f"P{int(entry['problem_id'])}_{entry['label']}"


def _build_worker_command(args: argparse.Namespace, entry: dict[str, Any], task_root: Path, gpu: str) -> list[str]:
    return [
        sys.executable,
        str(SINGLE_TASK_SCRIPT),
        "--kernelbench-root",
        args.kernelbench_root,
        "--output-root",
        str(task_root),
        "--gpu",
        gpu,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--warmup-runs",
        str(args.warmup_runs),
        "--profile-runs",
        str(args.profile_runs),
        "--only",
        str(entry["problem_id"]),
    ] + (["--skip-existing"] if args.skip_existing else [])


def _read_child_row(task_root: Path, entry: dict[str, Any], gpu: str, returncode: int, log_path: Path) -> dict[str, Any]:
    problem_id = int(entry["problem_id"])
    label = str(entry["label"])
    task_dir = task_root / _task_name(entry)
    profile_json = task_dir / "profile.json"
    if profile_json.exists():
        payload = json.loads(profile_json.read_text(encoding="utf-8"))
        profile = payload.get("profile") or {}
        return {
            "problem_id": problem_id,
            "label": label,
            "gpu": gpu,
            "status": payload.get("status", "unknown"),
            "best_speedup": entry.get("best_speedup"),
            "kernel_name": profile.get("kernel_name", ""),
            "row_count": profile.get("row_count", ""),
            "best_code": entry.get("best_code", ""),
            "profile_dir": str(task_dir.relative_to(REPO_ROOT)),
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "returncode": returncode,
            "error": profile.get("error") or payload.get("error", ""),
        }

    error = f"worker_exit_{returncode}"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            error = f"{error}: {lines[-1][:500]}"
    return {
        "problem_id": problem_id,
        "label": label,
        "gpu": gpu,
        "status": "worker_failed",
        "best_speedup": entry.get("best_speedup"),
        "kernel_name": "",
        "row_count": "",
        "best_code": entry.get("best_code", ""),
        "profile_dir": str(task_dir.relative_to(REPO_ROOT)),
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "returncode": returncode,
        "error": error,
    }


def _launch_worker(args: argparse.Namespace, entry: dict[str, Any], task_root: Path, gpu: str) -> tuple[subprocess.Popen[str], Path]:
    log_path = task_root / "worker.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDALLM_DEVICE"] = gpu
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    cuda_bin = str(Path(env["CUDA_HOME"]) / "bin")
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = os.pathsep.join([cuda_bin, python_bin_dir, "/usr/local/bin", "/usr/bin", "/bin", env.get("PATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        _build_worker_command(args, entry, task_root, gpu),
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._kw_log_handle = log_handle  # type: ignore[attr-defined]
    return process, log_path


def main() -> int:
    args = _parse_args()
    entries = _selected_entries(args.only)
    gpus = _parse_gpus(args.gpus)
    output_root = Path(args.output_root).resolve() if args.output_root else _default_output_root()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "entries": entries,
        "args": vars(args),
        "gpus": gpus,
        "single_task_script": str(SINGLE_TASK_SCRIPT),
    }
    _write_json(output_root / "manifest.json", manifest)

    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "task_count": len(entries), "gpus": gpus}, indent=2, ensure_ascii=False))
        return 0

    summary_path = output_root / "summary.tsv"
    summary_rows: list[dict[str, Any]] = []
    queue = list(entries)
    active: dict[str, dict[str, Any]] = {}

    print(f"[ncu-parallel] output_root={output_root}", flush=True)
    print(f"[ncu-parallel] tasks={len(entries)} gpus={','.join(gpus)}", flush=True)

    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            entry = queue.pop(0)
            task_root = output_root / _task_name(entry)
            task_root.mkdir(parents=True, exist_ok=True)
            process, log_path = _launch_worker(args, entry, task_root, gpu)
            active[gpu] = {
                "entry": entry,
                "task_root": task_root,
                "process": process,
                "log_path": log_path,
            }
            print(f"[ncu-parallel] launch gpu={gpu} task=P{entry['problem_id']} label={entry['label']} pid={process.pid}", flush=True)

        finished_gpus: list[str] = []
        for gpu, state in active.items():
            process: subprocess.Popen[str] = state["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            log_handle = getattr(process, "_kw_log_handle", None)
            if log_handle is not None:
                log_handle.close()
            row = _read_child_row(state["task_root"], state["entry"], gpu, returncode, state["log_path"])
            _append_summary(summary_path, row)
            summary_rows.append(row)
            print(f"[ncu-parallel] done gpu={gpu} task=P{row['problem_id']} status={row['status']} rc={returncode}", flush=True)
            finished_gpus.append(gpu)
        for gpu in finished_gpus:
            del active[gpu]

        if active or queue:
            time.sleep(args.poll_seconds)

    _write_json(output_root / "summary.json", summary_rows)
    success_count = sum(1 for row in summary_rows if row.get("status") == "ok")
    print(f"[ncu-parallel] complete success={success_count}/{len(summary_rows)} summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
