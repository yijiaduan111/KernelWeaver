"""Profile one validated saved CUDA candidate with Nsight Compute.

This is a diagnostic tool, not a search step. Its output describes the first
matching kernel launch for one test input, not every shape in KernelBench.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .resource_cost import phase_span


METRICS = (
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)
_METRIC_LINE = re.compile(r"^\s*(?P<name>(?:dram__throughput|sm__throughput|sm__warps_active)\.avg\.pct_of_peak_sustained_(?:elapsed|active))\s+%\s+(?P<value>[\d,.]+)\s*$", re.M)
_LAUNCH = re.compile(r"\b(?P<name>[A-Za-z_]\w*)(?:<[^;\n]*?>)?\s*<<<")


def classify_metrics(metrics: dict[str, float]) -> str:
    memory = metrics.get(METRICS[0])
    compute = metrics.get(METRICS[1])
    if memory is None or compute is None:
        return "unavailable"
    if memory >= 65 and memory >= 1.5 * compute:
        return "memory_pressure"
    if compute >= 65 and compute >= 1.5 * memory:
        return "compute_pressure"
    return "mixed_or_unresolved"


def strategy_hint(classification: str) -> str:
    return {
        "memory_pressure": "Investigate bytes moved and access pattern, including redundant reads, writes, and fusion.",
        "compute_pressure": "Investigate arithmetic per output, repeated index calculations, and reuse across neighboring outputs.",
        "mixed_or_unresolved": "Do not infer a single bottleneck from these counters; inspect the operator and collect more evidence.",
        "unavailable": "Counters unavailable; no profiler-guided strategy can be inferred.",
    }[classification]


def kernel_name(code: str) -> str | None:
    names = [m.group("name") for m in _LAUNCH.finditer(code)]
    return names[0] if names else None


def parse_metrics(text: str) -> dict[str, float]:
    found = {}
    for match in _METRIC_LINE.finditer(text):
        found[match.group("name")] = float(match.group("value").replace(",", ""))
    return found


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _worker(args) -> int:
    from ..core.loader import KernelBenchLoader
    from ..evaluation.evaluator_paper import KernelBenchPaperEvaluator
    from ..io import load_run

    run = load_run(args.run)
    node = run.nodes[args.node]
    task = KernelBenchLoader().load_official_problem(
        run.source_root, run.level, run.problem_id, backend=run.backend,
        semantics_enabled=run.config.semantics_enabled,
        semantics_mode=run.config.semantics_mode,
        semantics_max_anchor_hints=run.config.semantics_max_anchor_hints,
    )
    config = replace(run.config, num_correct_trials=max(3, run.config.num_correct_trials),
                     num_perf_trials=1, verbose=False)
    result = KernelBenchPaperEvaluator().evaluate(task, node.code, config)
    _write_json(Path(args.validation_output), {
        "correct": result.correct, "compile_ok": result.compile_ok,
        "failure_stage": result.failure_stage, "failure_type": result.failure_type,
    })
    return 0 if result.correct else 2


def _profile(args) -> int:
    from ..io import load_run

    run = load_run(args.run)
    if run.backend != "cuda" or args.node == "root":
        raise ValueError("Profile a non-root CUDA candidate")
    if args.node not in run.nodes or not run.nodes[args.node].correct:
        raise ValueError("Candidate must exist and have passed correctness")
    code = run.nodes[args.node].code
    name = kernel_name(code)
    if not name:
        raise ValueError("No CUDA kernel launch found in candidate")
    executable = shutil.which("ncu") or "/usr/local/cuda-12.2/bin/ncu"
    if not Path(executable).is_file():
        raise ValueError("Nsight Compute not installed")
    output = Path(args.output)
    validation_output = output.with_suffix(".validation.json")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    command = [executable, "--target-processes", "all", "--kernel-name-base", "function",
               "--kernel-name", f"regex:{name}", "--launch-count", "1",
               "--metrics", ",".join(METRICS), sys.executable, "-u", "-m",
               "src.experiment.profile_saved_kernel", "--worker", "--run", str(Path(args.run).resolve()),
               "--node", args.node, "--validation-output", str(validation_output.resolve())]
    # Count the real hardware-tool invocation, not merely an optional-feedback
    # gate which can return without launching ncu.  Logging is opt-in.
    with phase_span("profiling_ncu"):
        completed = subprocess.run(command, env=environment, capture_output=True, text=True,
                                   timeout=args.timeout_seconds, check=False)
    raw = completed.stdout + "\n" + completed.stderr
    log_path = output.with_suffix(".ncu.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(raw, encoding="utf-8")
    validation = json.loads(validation_output.read_text()) if validation_output.exists() else None
    metrics = parse_metrics(raw)
    classification = classify_metrics(metrics)
    payload = {
        "source_run": str(Path(args.run).resolve()), "node_id": args.node,
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "backend": run.backend, "task_name": run.task_name,
        "level": run.level, "problem_id": run.problem_id,
        "sampled_kernel": name, "gpu_index": args.gpu,
        "scope": "first matching kernel launch in one validation input",
        "metrics_percent_of_peak": metrics,
        "classification": classification,
        "strategy_hint": strategy_hint(classification),
        "validated_correct": bool(validation and validation["correct"]),
        "profiler_exit_code": completed.returncode,
        "log_path": str(log_path.resolve()),
    }
    _write_json(output, payload)
    print(json.dumps({k: payload[k] for k in ("task_name", "sampled_kernel", "classification",
        "metrics_percent_of_peak", "validated_correct")}, ensure_ascii=False))
    return 0 if completed.returncode == 0 and payload["validated_correct"] and metrics else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--output")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--validation-output")
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    if not args.output:
        parser.error("--output is required")
    return _profile(args)


if __name__ == "__main__":
    raise SystemExit(main())
