from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .schema import NcuProfile


_NCU_FALLBACK_PATH = "/usr/local/cuda-12.8/nsight-compute-2025.1.0/target/linux-desktop-glibc_2_11_3-x64/ncu"
_STARK_PYTHON = "/data/dyj/miniconda3/envs/stark/bin/python3"
_NCU_METRICS = [
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "gpu__time_duration.avg",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.ratio",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.ratio",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.max_rate",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.max_rate",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.max_rate",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.max_rate",
    "smsp__sass_branch_targets_threads_divergent.avg",
    "smsp__sass_branch_targets_threads_uniform.avg",
    "smsp__thread_inst_executed_pred_on_per_inst_executed.max_rate",
    "smsp__warps_eligible.avg",
]


def profile_candidate_with_ncu(
    task,
    candidate_code: str,
    *,
    timeout_seconds: int = 300,
    warmup_runs: int = 2,
    profile_runs: int = 3,
) -> tuple[NcuProfile, Path | None]:
    candidate_text = str(candidate_code or "")
    if not candidate_text.strip():
        return (
            NcuProfile(
                enabled=False,
                status="unsupported",
                notes=["Candidate source code is unavailable for diagnostics profiling."],
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

    metrics = list(_NCU_METRICS)
    with _temporary_profile_dir() as tmp_dir:
        tmp_path = Path(tmp_dir)
        candidate_module_path = tmp_path / "candidate_model.py"
        candidate_module_path.write_text(candidate_text, encoding="utf-8")
        reference_module_path = _materialize_reference_module(task, tmp_path)
        if reference_module_path is None:
            return (
                NcuProfile(
                    enabled=False,
                    status="unsupported",
                    notes=["Reference KernelBench program is unavailable for diagnostics profiling."],
                ),
                None,
            )

        script_path = tmp_path / "profile_target.py"
        csv_path = tmp_path / "ncu_raw.csv"
        python_bin = _resolve_python_bin()
        cuda_home = _resolve_cuda_home()
        torch_extensions_dir = tmp_path / "torch_extensions"
        script_path.write_text(
            _profile_script(
                reference_module_path=reference_module_path,
                candidate_module_path=candidate_module_path,
                source_root=str(getattr(task, "source_root", "") or ""),
                warmup_runs=warmup_runs,
                profile_runs=profile_runs,
                python_bin=python_bin,
                cuda_home=cuda_home,
                torch_extensions_dir=torch_extensions_dir,
            ),
            encoding="utf-8",
        )
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
            python_bin,
            "-B",
            str(script_path),
        ]
        env = _profile_subprocess_env(
            python_bin=python_bin,
            cuda_home=cuda_home,
            torch_extensions_dir=torch_extensions_dir,
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=str(tmp_path),
        )
        stdout = result.stdout or ""
        if result.returncode != 0 or not stdout.strip():
            return (
                NcuProfile(
                    enabled=False,
                    status="error",
                    profiler=Path(ncu_bin).name,
                    notes=["NCU profiling failed for the root candidate program."],
                    error=(result.stderr or stdout or f"ncu exited with code {result.returncode}").strip()[:1000],
                ),
                None,
            )

        csv_lines = [line for line in stdout.splitlines(keepends=True) if not line.startswith("==")]
        csv_path.write_text("".join(csv_lines), encoding="utf-8")
        persisted = Path(tempfile.mkdtemp(prefix="kw_diag_csv_")) / "ncu_raw.csv"
        shutil.copyfile(csv_path, persisted)
        kernel_name, row_count, raw_metrics = _select_dominant_kernel_metrics(persisted, metrics)
        if row_count <= 0:
            cleanup_profile_artifact(persisted)
            return (
                NcuProfile(
                    enabled=False,
                    status="error",
                    profiler=Path(ncu_bin).name,
                    notes=[
                        "NCU profiling failed for the root candidate program.",
                        "NCU CSV did not contain any kernel rows.",
                    ],
                    error="ncu_empty_csv",
                ),
                None,
            )
        return (
            NcuProfile(
                enabled=True,
                status="ok",
                profiler=Path(ncu_bin).name,
                kernel_name=kernel_name,
                row_count=row_count,
                kernel_launch_count=row_count,
                raw_metrics=raw_metrics,
                notes=["Profiled the root candidate program with Nsight Compute."],
            ),
            persisted,
        )


def _materialize_reference_module(task, workdir: Path) -> Path | None:
    source_origin = Path(str(getattr(task, "source_origin", "") or "")).resolve()
    if source_origin.exists():
        return source_origin
    reference_code = str(getattr(task, "reference_code", "") or "")
    if not reference_code.strip():
        return None
    reference_path = workdir / "reference_problem.py"
    reference_path.write_text(reference_code, encoding="utf-8")
    return reference_path


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


class _temporary_profile_dir:
    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix="kw_diag_")
        return self.path

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            shutil.rmtree(self.path, ignore_errors=True)
        except Exception:
            pass
        return False


def _resolve_ncu_bin() -> str | None:
    override = os.environ.get("KERNELWEAVER_NCU_BIN", "").strip()
    if override:
        return override if Path(override).exists() else shutil.which(override)
    if Path(_NCU_FALLBACK_PATH).exists():
        return _NCU_FALLBACK_PATH
    return shutil.which("ncu")


def _resolve_python_bin() -> str:
    if Path(_STARK_PYTHON).exists():
        return _STARK_PYTHON
    return sys.executable


def _resolve_cuda_home() -> str | None:
    candidates = [os.environ.get("CUDA_HOME", "").strip(), "/usr/local/cuda-12.8", "/usr/local/cuda"]
    for candidate in candidates:
        if candidate and Path(candidate, "bin", "nvcc").exists():
            return candidate
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    return None


def _prepend_env_path(env: dict[str, str], name: str, path: str | None) -> None:
    if not path:
        return
    current = env.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if path not in parts:
        env[name] = os.pathsep.join([path, *parts])


def _profile_subprocess_env(*, python_bin: str, cuda_home: str | None, torch_extensions_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions_dir))
    _prepend_env_path(env, "PATH", str(Path(python_bin).resolve().parent))
    if cuda_home:
        env.setdefault("CUDA_HOME", cuda_home)
        _prepend_env_path(env, "PATH", str(Path(cuda_home) / "bin"))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(Path(cuda_home) / "lib64"))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(Path(cuda_home) / "lib"))
    return env


def _select_dominant_kernel_metrics(csv_path: Path, metric_names: list[str]) -> tuple[str | None, int, dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, 0, {}

    duration_key = "gpu__time_duration.avg"

    def _duration(row: dict[str, str]) -> float:
        raw = str(row.get(duration_key, "")).strip().replace(",", "")
        try:
            return float(raw)
        except ValueError:
            return float("-inf")

    best_row = max(rows, key=_duration)
    raw_metrics: dict[str, Any] = {}
    for metric_name in metric_names:
        value = _parse_metric_value(best_row.get(metric_name))
        if value is not None:
            raw_metrics[metric_name] = value
    kernel_name = (best_row.get("Kernel Name") or "").strip() or None
    return kernel_name, len(rows), raw_metrics


def _parse_metric_value(value: Any) -> Any:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    normalized = text.replace(",", "")
    lowered = normalized.lower()
    if lowered in {"nan", "n/a", "none", "null", "--"}:
        return None
    try:
        number = float(normalized)
    except ValueError:
        return normalized
    if number.is_integer() and all(token not in normalized for token in (".", "e", "E")):
        return int(number)
    return number


def _profile_script(
    *,
    reference_module_path: Path,
    candidate_module_path: Path,
    source_root: str,
    warmup_runs: int,
    profile_runs: int,
    python_bin: str,
    cuda_home: str | None,
    torch_extensions_dir: Path,
) -> str:
    python_bin_dir = str(Path(python_bin).resolve().parent)
    cuda_home_text = str(cuda_home or "")
    return f"""
import importlib.util
import os
import sys
from pathlib import Path


def _prepend_env_path(name: str, value: str):
    if not value:
        return
    current = os.environ.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if value not in parts:
        os.environ[name] = os.pathsep.join([value, *parts])


_prepend_env_path("PATH", r"{python_bin_dir}")
if r"{cuda_home_text}":
    os.environ.setdefault("CUDA_HOME", r"{cuda_home_text}")
    _prepend_env_path("PATH", str(Path(r"{cuda_home_text}") / "bin"))
    _prepend_env_path("LD_LIBRARY_PATH", str(Path(r"{cuda_home_text}") / "lib64"))
    _prepend_env_path("LD_LIBRARY_PATH", str(Path(r"{cuda_home_text}") / "lib"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", r"{str(torch_extensions_dir)}")
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

import torch


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def _resolve_callable(module, name: str):
    fn = getattr(module, name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"module '{{module.__name__}}' is missing callable {{name}}")
    return fn


def _resolve_model_cls(module):
    for name in ("ModelNew", "Model"):
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise AttributeError(f"module '{{module.__name__}}' is missing ModelNew/Model class")


def _instantiate_model(model_cls, init_inputs):
    init_inputs = _to_cuda(init_inputs)
    if isinstance(init_inputs, dict):
        return model_cls(**init_inputs).cuda().eval()
    if isinstance(init_inputs, tuple):
        return model_cls(*init_inputs).cuda().eval()
    if isinstance(init_inputs, list):
        return model_cls(*init_inputs).cuda().eval()
    return model_cls(init_inputs).cuda().eval()


def _call_model(model, inputs):
    inputs = _to_cuda(inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, list):
        return model(*inputs)
    return model(inputs)


source_root_text = r"{source_root}"
if source_root_text:
    source_root = Path(source_root_text)
    for candidate in (source_root / "src", source_root):
        if candidate.exists():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)

reference_module = _load_module(Path(r"{str(reference_module_path)}"), "kb_reference")
candidate_module = _load_module(Path(r"{str(candidate_module_path)}"), "kb_candidate")
get_init_inputs = _resolve_callable(reference_module, "get_init_inputs")
get_inputs = _resolve_callable(reference_module, "get_inputs")
model_cls = _resolve_model_cls(candidate_module)
model = _instantiate_model(model_cls, get_init_inputs())
inputs = get_inputs()
with torch.no_grad():
    for _ in range({int(warmup_runs)}):
        _call_model(model, inputs)
    torch.cuda.synchronize()
    for _ in range({int(profile_runs)}):
        _call_model(model, inputs)
    torch.cuda.synchronize()
""".strip() + "\n"
