"""Collect optional hardware feedback after one validated candidate.

This module knows the profiling adapter; agent context receives backend-neutral
pressure observations, not CUDA instructions or a guaranteed task diagnosis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from ..io import save_run
from ..models import RunResult
from ..utils import normalized_code_hash


def profile_live_candidate(task, config, node, stats: dict) -> dict:
    from ..experiment.profile_saved_kernel import _profile

    if not task.source_root:
        stats["profile_feedback_status"] = "missing_source_root"
        return {}
    output_root = Path(config.long_term_memory_path).parent if config.long_term_memory_path else None
    try:
        with TemporaryDirectory(prefix="pyramidkernel_profile_") as temporary:
            temporary_path = Path(temporary)
            run = RunResult(
                task_name=task.name, config=config, best_node_id=node.node_id,
                leaderboard=[node.node_id], nodes={node.node_id: node},
                selection_history=[], stats={}, benchmark_family=task.benchmark_family,
                level=task.level, problem_id=task.problem_id, backend=task.backend,
                source_root=task.source_root,
            )
            run_path = save_run(run, temporary_path)
            output = ((output_root / "profiles" / f"{task.level}_{task.problem_id}_{node.node_id}.json")
                      if output_root else temporary_path / "profile.json")
            args = SimpleNamespace(
                run=str(run_path), node=node.node_id, output=str(output),
                gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                timeout_seconds=config.profile_feedback_timeout_seconds,
            )
            status = _profile(args)
            if status:
                stats["profile_feedback_status"] = "profile_failed"
                return {}
            payload = json.loads(output.read_text(encoding="utf-8"))
            if output_root:
                # The temporary run is deleted after profiling; keep the durable
                # artifact tied to the candidate's own code hash instead.
                payload["source_run"] = "online_candidate"
                payload["observed_code_hash"] = normalized_code_hash(node.code)
                output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        stats["profile_feedback_status"] = type(exc).__name__
        return {}
    if not payload.get("validated_correct") or payload.get("classification") == "unavailable":
        stats["profile_feedback_status"] = "unusable_profile"
        return {}
    feedback = {
        "observed_node_id": node.node_id,
        "observed_code_hash": normalized_code_hash(node.code),
        "backend": task.backend,
        "scope": payload["scope"],
        "sampled_kernel": payload["sampled_kernel"],
        "pressure_class": payload["classification"],
        "metrics_percent_of_peak": payload["metrics_percent_of_peak"],
        "strategy_hint": payload["strategy_hint"],
    }
    stats["profile_feedback_status"] = "available"
    stats["profile_feedback"] = feedback
    return feedback
