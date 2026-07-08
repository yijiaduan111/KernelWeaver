from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.loader import KernelBenchLoader
from src.diagnostics.ncu import cleanup_profile_artifact, profile_candidate_with_ncu


BEST_CODE_MANIFEST: list[dict[str, Any]] = [
    {
        "problem_id": 1,
        "label": "P1_SquareMatmul",
        "best_speedup": 1.415,
        "best_code": "runs/sequential_main_l1_15_claude_cuda_main_20260703_160343_continuefix/01_L1_P1_SquareMatmul_P1_20260703_160343/best_code.py",
    },
    {
        "problem_id": 10,
        "label": "P10_TensorMatmul3D",
        "best_speedup": 1.548,
        "best_code": "runs/retry_failed_main_l1_15_claude_cuda_main_4way_20260707_125933/02_L1_P10_TensorMatmul3D_P10/best_code.py",
    },
    {
        "problem_id": 20,
        "label": "P20_LeakyReLU",
        "best_speedup": 1.012,
        "best_code": "runs/sequential_main_l1_15_main_cuda_20260704_192908/03_L1_P20_LeakyReLU_P20_20260704_205030/best_code.py",
    },
    {
        "problem_id": 25,
        "label": "P25_Swish",
        "best_speedup": 2.345,
        "best_code": "runs/sequential_main_l1_15_main_cuda_20260704_192908/04_L1_P25_Swish_P25_20260704_222012/best_code.py",
    },
    {
        "problem_id": 33,
        "label": "P33_BatchNorm",
        "best_speedup": 1.011,
        "best_code": "runs/main_l1_15_cuda_claude_delib_main_a10_6x1_20260525_172639/group_1/L1_P33_BatchNorm_l1_p33/best_code.py",
    },
    {
        "problem_id": 40,
        "label": "P40_LayerNorm",
        "best_speedup": 5.558,
        "best_code": "runs/sequential_main_l1_15_claude_cuda_main_20260704_001434_notool/06_L1_P40_LayerNorm_P40_20260704_011323/best_code.py",
    },
    {
        "problem_id": 42,
        "label": "P42_MaxPool2d",
        "best_speedup": 1.973,
        "best_code": "runs/main_l1_15_cuda_claude_delib_main_a30_4x2_20260525_010516/group_1/L1_P42_MaxPool2d_l1_p42/best_code.py",
    },
    {
        "problem_id": 45,
        "label": "P45_AvgPool2d",
        "best_speedup": 1.088,
        "best_code": "runs/main_l1_15_claude_cuda_main_4way_20260707_001550/08_L1_P45_AvgPool2d_P45/best_code.py",
    },
    {
        "problem_id": 47,
        "label": "P47_SumReduction",
        "best_speedup": 1.123,
        "best_code": "runs/retry_failed_main_l1_15_claude_cuda_main_4way_20260707_125933/09_L1_P47_SumReduction_P47/best_code.py",
    },
    {
        "problem_id": 89,
        "label": "P89_Cumsum",
        "best_speedup": 1.174,
        "best_code": "runs/sequential_main_l1_15_main_cuda_20260704_192908/10_L1_P89_Cumsum_P89_20260705_013002/best_code.py",
    },
    {
        "problem_id": 50,
        "label": "P50_Conv2dStandard",
        "best_speedup": 1.169,
        "best_code": "runs/sequential_main_l1_15_main_cuda_20260704_192908/11_L1_P50_Conv2dStandard_P50_20260705_024057/best_code.py",
    },
    {
        "problem_id": 61,
        "label": "P61_ConvTranspose3d",
        "best_speedup": 1.000,
        "best_code": "runs/sequential_main_l1_15_main_cuda_20260704_192908/12_L1_P61_ConvTranspose3d_P61_20260705_030746/best_code.py",
    },
    {
        "problem_id": 82,
        "label": "P82_DepthwiseConv2d",
        "best_speedup": 1.233,
        "best_code": "runs/retry_failed_main_l1_15_claude_cuda_main_4way_20260707_125933/13_L1_P82_DepthwiseConv2d_P82/best_code.py",
    },
    {
        "problem_id": 95,
        "label": "P95_CrossEntropyLoss",
        "best_speedup": None,
        "best_code": None,
        "skip_reason": "no_valid_best_code_found",
    },
    {
        "problem_id": 97,
        "label": "P97_ScaledDotProductAttention",
        "best_speedup": 1.154,
        "best_code": "runs/retry_failed_main_l1_15_claude_cuda_main_20260706_125030/05_L1_P97_ScaledDotProductAttention_P97_20260706_153130/best_code.py",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nsight Compute on curated KernelWeaver best_code.py files.")
    parser.add_argument("--kernelbench-root", default="KernelBench", help="KernelBench root under the repo or absolute path.")
    parser.add_argument("--output-root", default=None, help="Output directory. Defaults to runs/ncu_best_codes_<timestamp>.")
    parser.add_argument("--gpu", default=None, help="Single CUDA device id to expose, e.g. 0.")
    parser.add_argument("--timeout-seconds", type=int, default=1200, help="Per-task NCU timeout.")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--profile-runs", type=int, default=3)
    parser.add_argument("--only", default="", help="Comma-separated problem ids, e.g. 1,25,40.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tasks whose profile.json already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and exit without running NCU.")
    return parser.parse_args()


def _selected_entries(only: str) -> list[dict[str, Any]]:
    if not only.strip():
        return list(BEST_CODE_MANIFEST)
    wanted = {int(item.strip().lstrip("Pp")) for item in only.split(",") if item.strip()}
    return [entry for entry in BEST_CODE_MANIFEST if int(entry["problem_id"]) in wanted]


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"ncu_best_codes_{stamp}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _append_summary(path: Path, row: dict[str, Any]) -> None:
    header = [
        "problem_id",
        "label",
        "status",
        "best_speedup",
        "kernel_name",
        "row_count",
        "best_code",
        "profile_dir",
        "error",
    ]
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write("\t".join(header) + "\n")
        handle.write("\t".join(str(row.get(key, "") or "") for key in header) + "\n")


def _profile_one(entry: dict[str, Any], args: argparse.Namespace, output_root: Path, loader: KernelBenchLoader) -> dict[str, Any]:
    problem_id = int(entry["problem_id"])
    label = str(entry["label"])
    task_dir = output_root / f"P{problem_id}_{label}"
    profile_json = task_dir / "profile.json"
    task_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and profile_json.exists():
        return {
            "problem_id": problem_id,
            "label": label,
            "status": "skipped_existing",
            "best_speedup": entry.get("best_speedup"),
            "best_code": entry.get("best_code"),
            "profile_dir": str(task_dir.relative_to(REPO_ROOT)),
        }

    best_code_rel = entry.get("best_code")
    if not best_code_rel:
        payload = {"entry": entry, "status": "missing_best_code", "error": entry.get("skip_reason", "missing_best_code")}
        _write_json(profile_json, payload)
        return {
            "problem_id": problem_id,
            "label": label,
            "status": "missing_best_code",
            "best_speedup": entry.get("best_speedup"),
            "best_code": "",
            "profile_dir": str(task_dir.relative_to(REPO_ROOT)),
            "error": entry.get("skip_reason", "missing_best_code"),
        }

    best_code_path = (REPO_ROOT / str(best_code_rel)).resolve()
    if not best_code_path.exists():
        payload = {"entry": entry, "status": "missing_best_code_file", "error": str(best_code_path)}
        _write_json(profile_json, payload)
        return {
            "problem_id": problem_id,
            "label": label,
            "status": "missing_best_code_file",
            "best_speedup": entry.get("best_speedup"),
            "best_code": best_code_rel,
            "profile_dir": str(task_dir.relative_to(REPO_ROOT)),
            "error": str(best_code_path),
        }

    candidate_code = best_code_path.read_text(encoding="utf-8")
    task = loader.load_official_problem(args.kernelbench_root, 1, problem_id, backend="cuda", semantics_enabled=False)
    csv_path = None
    try:
        profile, csv_path = profile_candidate_with_ncu(
            task,
            candidate_code,
            timeout_seconds=args.timeout_seconds,
            warmup_runs=args.warmup_runs,
            profile_runs=args.profile_runs,
        )
        copied_csv = None
        if csv_path is not None and csv_path.exists():
            copied_csv = task_dir / "ncu_raw.csv"
            shutil.copyfile(csv_path, copied_csv)
        profile_payload = asdict(profile)
        payload = {
            "entry": entry,
            "status": profile.status,
            "profile": profile_payload,
            "best_code_abs": str(best_code_path),
            "ncu_raw_csv": str(copied_csv.relative_to(REPO_ROOT)) if copied_csv else None,
        }
        _write_json(profile_json, payload)
        return {
            "problem_id": problem_id,
            "label": label,
            "status": profile.status,
            "best_speedup": entry.get("best_speedup"),
            "kernel_name": profile.kernel_name or "",
            "row_count": profile.row_count,
            "best_code": best_code_rel,
            "profile_dir": str(task_dir.relative_to(REPO_ROOT)),
            "error": profile.error or "",
        }
    except Exception as exc:  # noqa: BLE001 - batch profiler must continue across tasks.
        payload = {"entry": entry, "status": "exception", "best_code_abs": str(best_code_path), "error": repr(exc)}
        _write_json(profile_json, payload)
        return {
            "problem_id": problem_id,
            "label": label,
            "status": "exception",
            "best_speedup": entry.get("best_speedup"),
            "best_code": best_code_rel,
            "profile_dir": str(task_dir.relative_to(REPO_ROOT)),
            "error": repr(exc),
        }
    finally:
        cleanup_profile_artifact(csv_path)


def main() -> int:
    args = _parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDALLM_DEVICE", str(args.gpu))
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    cuda_bin = str(Path(os.environ["CUDA_HOME"]) / "bin")
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join([cuda_bin, python_bin_dir, os.environ.get("PATH", "")])

    output_root = Path(args.output_root).resolve() if args.output_root else _default_output_root()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    entries = _selected_entries(args.only)
    manifest_path = output_root / "manifest.json"
    summary_path = output_root / "summary.tsv"
    _write_json(manifest_path, {"created_at": datetime.now().isoformat(), "entries": entries, "args": vars(args)})

    missing = []
    for entry in entries:
        best_code = entry.get("best_code")
        if best_code and not (REPO_ROOT / str(best_code)).exists():
            missing.append({"problem_id": entry["problem_id"], "best_code": best_code})
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "tasks": len(entries), "missing_files": missing}, indent=2, ensure_ascii=False))
        return 1 if missing else 0

    print(f"[ncu-batch] output_root={output_root}", flush=True)
    print(f"[ncu-batch] tasks={len(entries)} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}", flush=True)
    if missing:
        print(f"[ncu-batch] warning missing_files={missing}", flush=True)

    loader = KernelBenchLoader()
    completed = []
    for index, entry in enumerate(entries, start=1):
        print(f"[ncu-batch] {index}/{len(entries)} start P{entry['problem_id']} {entry['label']}", flush=True)
        row = _profile_one(entry, args, output_root, loader)
        _append_summary(summary_path, row)
        completed.append(row)
        print(f"[ncu-batch] {index}/{len(entries)} done P{entry['problem_id']} status={row.get('status')} kernel={row.get('kernel_name', '')}", flush=True)

    _write_json(output_root / "summary.json", completed)
    print(f"[ncu-batch] complete summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())