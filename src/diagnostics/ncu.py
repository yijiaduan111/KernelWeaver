from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..km_machine_check import load_yaml_rules
from .schema import NcuProfile


DEFAULT_RULEBOOK_PATH = Path(__file__).resolve().parents[1] / "km_bottleneck.yaml"
_NCU_FALLBACK_PATH = "/usr/local/cuda-12.8/nsight-compute-2025.1.0/target/linux-desktop-glibc_2_11_3-x64/ncu"
_STARK_PYTHON = "/data/dyj/miniconda3/envs/stark/bin/python3"


def profile_reference_with_ncu(
    task,
    *,
    timeout_seconds: int = 300,
    warmup_runs: int = 2,
    profile_runs: int = 3,
) -> tuple[NcuProfile, Path | None]:
    source_origin = Path(str(getattr(task, "source_origin", "") or "")).resolve()
    if not source_origin.exists():
        return (
            NcuProfile(
                enabled=False,
                status="unsupported",
                notes=["Reference source path is unavailable for diagnostics profiling."],
            ),
            None,
        )
    ncu_bin = _resolve_ncu_bin()
    if ncu_bin is None:
        return (
            NcuProfile(
                enabled=False,
                status="unavailable",
                notes=["NCU binary was not found in PATH."],
                error="ncu_not_found",
            ),
            None,
        )
    with tempfile.TemporaryDirectory(prefix="kw_diag_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        script_path = tmp_path / "profile_reference.py"
        csv_path = tmp_path / "ncu_raw.csv"
        script_path.write_text(_profile_script(source_origin, warmup_runs, profile_runs), encoding="utf-8")
        metrics = _metric_names_from_rulebook(DEFAULT_RULEBOOK_PATH)
        cmd = [
            "sudo",
            ncu_bin,
            "--target-processes",
            "all",
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
            "--metrics",
            ",".join(metrics),
            _STARK_PYTHON if Path(_STARK_PYTHON).exists() else sys.executable,
            str(script_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ.copy(),
            cwd=str(tmp_path),
        )
        stdout = result.stdout or ""
        if result.returncode != 0 or not stdout.strip():
            return (
                NcuProfile(
                    enabled=False,
                    status="error",
                    profiler=Path(ncu_bin).name,
                    notes=["NCU profiling failed for the reference program."],
                    error=(result.stderr or stdout or f"ncu exited with code {result.returncode}").strip()[:1000],
                ),
                None,
            )
        csv_lines = [l for l in stdout.splitlines(keepends=True) if not l.startswith("==")]
        csv_path.write_text("".join(csv_lines), encoding="utf-8")
        persisted = Path(tempfile.mkdtemp(prefix="kw_diag_csv_")) / "ncu_raw.csv"
        shutil.copyfile(csv_path, persisted)
        kernel_name, row_count = _select_dominant_kernel(persisted)
        return (
            NcuProfile(
                enabled=True,
                status="ok",
                profiler=Path(ncu_bin).name,
                kernel_name=kernel_name,
                row_count=row_count,
                kernel_launch_count=row_count,
                notes=["Profiled the reference KernelBench program with Nsight Compute."],
            ),
            persisted,
        )


def cleanup_profile_artifact(csv_path: Path | None) -> None:
    if csv_path is None:
        return
    try:
        if csv_path.exists():
            csv_path.unlink()
        parent = csv_path.parent
        if parent.exists() and parent.name.startswith("kw_diag_csv_"):
            parent.rmdir()
    except OSError:
        return


def _resolve_ncu_bin() -> str | None:
    override = os.environ.get("KERNELWEAVER_NCU_BIN", "").strip()
    if override:
        return override if Path(override).exists() else shutil.which(override)
    if Path(_NCU_FALLBACK_PATH).exists():
        return _NCU_FALLBACK_PATH
    return shutil.which("ncu")


def _metric_names_from_rulebook(rulebook_path: Path) -> list[str]:
    payload = load_yaml_rules(rulebook_path)
    mapping = ((payload.get("machine_check") or {}).get("input_normalization") or {}).get("field_mapping") or {}
    names: list[str] = []
    for value in mapping.values():
        name = str(value).strip()
        if name and name not in names:
            names.append(name)
    return names


def _select_dominant_kernel(csv_path: Path) -> tuple[str | None, int]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, 0
    duration_key = "gpu__time_duration.avg"
    def _duration(row: dict[str, str]) -> float:
        raw = str(row.get(duration_key, "")).strip().replace(",", "")
        try:
            return float(raw)
        except ValueError:
            return float("-inf")
    best_row = max(rows, key=_duration)
    return (best_row.get("Kernel Name") or "").strip() or None, len(rows)


def _profile_script(problem_path: Path, warmup_runs: int, profile_runs: int) -> str:
    return f"""
import importlib.util
from pathlib import Path
import torch

def _to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, list):
        return [_to_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cuda(item) for item in value)
    if isinstance(value, dict):
        return {{key: _to_cuda(item) for key, item in value.items()}}
    return value

problem_path = Path(r\"{str(problem_path)}\")
spec = importlib.util.spec_from_file_location("kb_problem", problem_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
model = module.Model(*_to_cuda(module.get_init_inputs())).cuda().eval()
inputs = _to_cuda(module.get_inputs())
with torch.no_grad():
    for _ in range({int(warmup_runs)}):
        model(*inputs)
    torch.cuda.synchronize()
    for _ in range({int(profile_runs)}):
        model(*inputs)
    torch.cuda.synchronize()
""".strip() + "\n"
