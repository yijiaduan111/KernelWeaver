"""Persistence helpers for STARK run artifacts.

Saved runs are the contract between execution, replay, validation, and
reporting. Backward compatibility matters here, so loading code keeps
default fallbacks for older result files whenever new fields are added.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, fields
from pathlib import Path

from .config import (
    legacy_evaluation_profile_name,
    legacy_kernelbench_evaluator_name,
    legacy_preset_name,
    resolve_evaluator_profile,
    resolve_measurement_profile,
    resolve_run_profile,
    resolve_search_profile,
)
from .deliberation import strategy_portfolio_from_dict, strategy_portfolio_to_dict
from .feedback import feedback_state_from_dict, feedback_state_to_dict
from .models import AnchorEdit, GroundedRegion, RunResult, SearchNode, StarkConfig
from .semantics import semantic_profile_from_dict, semantic_profile_to_dict


def _dump_float(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isfinite(value):
        return value
    return None


def _infer_failure_stage(failure_type: str | None) -> str | None:
    if failure_type is None:
        return None
    if "compile" in failure_type or failure_type == "reference_error":
        return "compile"
    if "runtime" in failure_type:
        return "runtime"
    if "correctness" in failure_type:
        return "correctness"
    return None


def _infer_node_status(node_payload: dict) -> str:
    if node_payload.get("node_status"):
        return node_payload["node_status"]
    if node_payload.get("prune_reason"):
        return "pruned"
    if node_payload.get("compile_ok") and node_payload.get("correct"):
        return "correct"
    failure_stage = node_payload.get("latest_failure_stage") or _infer_failure_stage(node_payload.get("failure_type"))
    if failure_stage == "compile":
        return "compile_fail"
    if failure_stage == "runtime":
        return "runtime_fail"
    if failure_stage == "correctness":
        return "correctness_fail"
    return "correct"


def save_run(run: RunResult, output_dir: str | Path) -> Path:
    """Persist one run to `run.json` and write the current best code beside it."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": run.task_name,
        "config": asdict(run.config),
        "best_node_id": run.best_node_id,
        "leaderboard": list(run.leaderboard),
        "leaderboard_history": [list(item) for item in run.leaderboard_history],
        "selection_history": list(run.selection_history),
        "selection_reasons": list(run.selection_reasons),
        "selection_exclusions": [dict(item) for item in run.selection_exclusions],
        "pruned_nodes": dict(run.pruned_nodes),
        "debug_stats": run.debug_stats,
        "stats": run.stats,
        "benchmark_family": run.benchmark_family,
        "level": run.level,
        "problem_id": run.problem_id,
        "backend": run.backend,
        "source_origin": run.source_origin,
        "source_root": run.source_root,
        "workflow": run.workflow,
        "run_profile": run.run_profile,
        "search_profile": run.search_profile,
        "evaluator_profile": run.evaluator_profile,
        "measurement_profile": run.measurement_profile,
        "preset": run.preset,
        "evaluation_profile": run.evaluation_profile,
        "kernelbench_evaluator": run.kernelbench_evaluator,
        "reference_runtimes": {key: _dump_float(value) for key, value in run.reference_runtimes.items()},
        "speedups": {key: _dump_float(value) for key, value in run.speedups.items()},
        "primary_reference": run.primary_reference,
        "semantic_profile": semantic_profile_to_dict(run.semantic_profile),
        "strategy_portfolio": strategy_portfolio_to_dict(run.strategy_portfolio),
        "feedback_state": feedback_state_to_dict(run.feedback_state),
        "grounded_regions": [
            {
                "anchor_name": region.anchor_name,
                "region_role": region.region_role,
                "start_line": region.start_line,
                "end_line": region.end_line,
                "source_excerpt": region.source_excerpt,
                "source_hash": region.source_hash,
            }
            for region in run.grounded_regions
        ],
        "nodes": {
            node_id: {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "code": node.code,
                "origin": node.origin,
                "child_ids": list(node.child_ids),
                "selected_count": node.selected_count,
                "plan_strategy_name": node.plan_strategy_name,
                "plan_summary": node.plan_summary,
                "anchor_edits": [
                    {
                        "anchor_name": edit.anchor_name,
                        "instruction": edit.instruction,
                        "operation": edit.operation,
                    }
                    for edit in node.anchor_edits
                ],
                "compile_ok": node.compile_ok,
                "correct": node.correct,
                "runtime": _dump_float(node.runtime),
                "score": _dump_float(node.score),
                "logs": list(node.logs),
                "failure_type": node.failure_type,
                "node_status": node.node_status,
                "selection_reason": node.selection_reason,
                "prune_reason": node.prune_reason,
                "debug_attempts": node.debug_attempts,
                "latest_failure_stage": node.latest_failure_stage,
                "reference_runtime": _dump_float(node.reference_runtime),
                "speedup": _dump_float(node.speedup),
                "reference_runtimes": {key: _dump_float(value) for key, value in node.reference_runtimes.items()},
                "speedups": {key: _dump_float(value) for key, value in node.speedups.items()},
                "primary_reference": node.primary_reference,
                "plan_mode": node.plan_mode,
                "performance_hypothesis": node.performance_hypothesis,
                "single_change_focus": node.single_change_focus,
                "mutation_family": node.mutation_family,
                "target_metric": node.target_metric,
            }
            for node_id, node in run.nodes.items()
        },
    }
    run_path = target_dir / "run.json"
    run_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (target_dir / "best_code.py").write_text(run.nodes[run.best_node_id].code, encoding="utf-8")
    return run_path


def _load_config(config_payload: dict) -> StarkConfig:
    allowed = {field.name for field in fields(StarkConfig)}
    filtered = {key: value for key, value in config_payload.items() if key in allowed}
    return StarkConfig(**filtered)


def load_run(path: str | Path) -> RunResult:
    """Load a saved run while remaining tolerant of older artifact formats."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {}
    for node_id, node_payload in payload["nodes"].items():
        node_reference_runtimes = dict(node_payload.get("reference_runtimes") or {})
        if not node_reference_runtimes and node_payload.get("reference_runtime") is not None:
            node_reference_runtimes["torch_eager"] = node_payload.get("reference_runtime")
        node_speedups = dict(node_payload.get("speedups") or {})
        if not node_speedups and node_payload.get("speedup") is not None:
            node_speedups["torch_eager"] = node_payload.get("speedup")
        nodes[node_id] = SearchNode(
            node_id=node_payload["node_id"],
            parent_id=node_payload.get("parent_id"),
            depth=node_payload["depth"],
            code=node_payload["code"],
            origin=node_payload.get("origin", "plan_code"),
            child_ids=list(node_payload.get("child_ids", [])),
            selected_count=node_payload.get("selected_count", 0),
            plan_strategy_name=node_payload.get("plan_strategy_name"),
            plan_summary=node_payload.get("plan_summary"),
            anchor_edits=[
                AnchorEdit(
                    anchor_name=edit["anchor_name"],
                    instruction=edit["instruction"],
                    operation=edit.get("operation", "replace"),
                )
                for edit in node_payload.get("anchor_edits", [])
            ],
            compile_ok=node_payload.get("compile_ok", False),
            correct=node_payload.get("correct", False),
            runtime=node_payload.get("runtime"),
            score=node_payload["score"] if node_payload.get("score") is not None else float("inf"),
            logs=list(node_payload.get("logs", [])),
            failure_type=node_payload.get("failure_type"),
            node_status=_infer_node_status(node_payload),
            selection_reason=node_payload.get("selection_reason"),
            prune_reason=node_payload.get("prune_reason"),
            debug_attempts=node_payload.get("debug_attempts", 0),
            latest_failure_stage=node_payload.get("latest_failure_stage") or _infer_failure_stage(node_payload.get("failure_type")),
            reference_runtime=node_payload.get("reference_runtime"),
            speedup=node_payload.get("speedup"),
            reference_runtimes=node_reference_runtimes,
            speedups=node_speedups,
            primary_reference=node_payload.get("primary_reference", "torch_eager" if node_reference_runtimes else None),
            plan_mode=node_payload.get("plan_mode", "explore"),
            performance_hypothesis=node_payload.get("performance_hypothesis"),
            single_change_focus=node_payload.get("single_change_focus"),
            mutation_family=node_payload.get("mutation_family"),
            target_metric=node_payload.get("target_metric"),
        )
    leaderboard = list(payload.get("leaderboard", []))
    leaderboard_history = payload.get("leaderboard_history") or [leaderboard]
    selection_history = list(payload.get("selection_history", []))
    selection_reasons = list(payload.get("selection_reasons", []))
    if not selection_reasons:
        selection_reasons = ["fallback_no_finite_score"] * len(selection_history)
    selection_exclusions = [dict(item) for item in payload.get("selection_exclusions", [])]
    if not selection_exclusions:
        selection_exclusions = [{} for _ in selection_history]
    debug_stats = payload.get("debug_stats") or {
        "total_attempts": payload.get("stats", {}).get("debug_attempts", 0),
        "per_node": {},
    }
    run_reference_runtimes = dict(payload.get("reference_runtimes") or {})
    if not run_reference_runtimes and payload.get("best_node_id") in nodes:
        run_reference_runtimes = dict(nodes[payload["best_node_id"]].reference_runtimes)
    run_speedups = dict(payload.get("speedups") or {})
    if not run_speedups and payload.get("best_node_id") in nodes:
        run_speedups = dict(nodes[payload["best_node_id"]].speedups)
    config_payload = payload.get("config", {})
    config = _load_config(config_payload)
    legacy_preset = payload.get("preset", config_payload.get("preset", getattr(config, "preset", "default")))
    legacy_measurement = payload.get("evaluation_profile", config_payload.get("evaluation_profile", getattr(config, "evaluation_profile", "kernelbench_reduced_v1")))
    legacy_evaluator = payload.get("kernelbench_evaluator", config_payload.get("kernelbench_evaluator", getattr(config, "kernelbench_evaluator", "paper")))

    run_profile_present = "run_profile" in payload or "run_profile" in config_payload
    raw_run_profile = payload["run_profile"] if "run_profile" in payload else config_payload.get("run_profile")
    run_profile = raw_run_profile if run_profile_present else resolve_run_profile(None, legacy_preset)

    search_profile_present = "search_profile" in payload or "search_profile" in config_payload
    raw_search_profile = payload["search_profile"] if "search_profile" in payload else config_payload.get("search_profile")
    search_profile = raw_search_profile if search_profile_present else resolve_search_profile(None, run_profile or resolve_run_profile(None, legacy_preset), legacy_preset)

    evaluator_profile_present = "evaluator_profile" in payload or "evaluator_profile" in config_payload
    raw_evaluator_profile = payload["evaluator_profile"] if "evaluator_profile" in payload else config_payload.get("evaluator_profile")
    evaluator_profile = raw_evaluator_profile if evaluator_profile_present else resolve_evaluator_profile(None, run_profile or resolve_run_profile(None, legacy_preset), legacy_evaluator)

    measurement_profile_present = "measurement_profile" in payload or "measurement_profile" in config_payload
    raw_measurement_profile = payload["measurement_profile"] if "measurement_profile" in payload else config_payload.get("measurement_profile")
    measurement_profile = raw_measurement_profile if measurement_profile_present else resolve_measurement_profile(None, run_profile or resolve_run_profile(None, legacy_preset), legacy_measurement)

    return RunResult(
        task_name=payload["task_name"],
        config=config,
        best_node_id=payload["best_node_id"],
        leaderboard=leaderboard,
        nodes=nodes,
        selection_history=selection_history,
        stats=payload.get("stats", {}),
        leaderboard_history=[list(item) for item in leaderboard_history],
        selection_reasons=selection_reasons,
        selection_exclusions=selection_exclusions,
        pruned_nodes=dict(payload.get("pruned_nodes", {})),
        debug_stats=debug_stats,
        benchmark_family=payload.get("benchmark_family"),
        level=payload.get("level"),
        problem_id=payload.get("problem_id"),
        backend=payload.get("backend"),
        source_origin=payload.get("source_origin"),
        source_root=payload.get("source_root"),
        workflow=payload.get("workflow", "stark"),
        run_profile=run_profile,
        search_profile=search_profile,
        evaluator_profile=evaluator_profile,
        measurement_profile=measurement_profile,
        preset=payload.get("preset", payload.get("config", {}).get("preset", legacy_preset_name(search_profile))),
        evaluation_profile=payload.get("evaluation_profile", payload.get("config", {}).get("evaluation_profile", legacy_evaluation_profile_name(measurement_profile))),
        kernelbench_evaluator=payload.get("kernelbench_evaluator", payload.get("config", {}).get("kernelbench_evaluator", legacy_kernelbench_evaluator_name(evaluator_profile))),
        reference_runtimes=run_reference_runtimes,
        speedups=run_speedups,
        primary_reference=payload.get("primary_reference", "torch_eager" if run_reference_runtimes else None),
        semantic_profile=semantic_profile_from_dict(payload.get("semantic_profile")),
        strategy_portfolio=strategy_portfolio_from_dict(payload.get("strategy_portfolio")),
        feedback_state=feedback_state_from_dict(payload.get("feedback_state")),
        grounded_regions=[
            GroundedRegion(
                anchor_name=region["anchor_name"],
                region_role=region.get("region_role", region["anchor_name"]),
                start_line=int(region.get("start_line", 0)),
                end_line=int(region.get("end_line", 0)),
                source_excerpt=region.get("source_excerpt", ""),
                source_hash=region.get("source_hash", ""),
            )
            for region in payload.get("grounded_regions", [])
        ],
    )
