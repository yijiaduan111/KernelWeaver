"""Layered YAML config loader for the KernelWeaver baseline."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _REPO_ROOT / "configs"

_WORKFLOWS = ["stark", "sampling", "reflexion", "search-agent", "ma-only"]
_BACKENDS = ["triton", "cuda"]

_DEFAULTS: dict[str, Any] = {
    "workflow": "stark",
    "backend": "triton",
    "provider": "mock",
    "run_profile": "quick_local",
    "agent_provider_profile": "all_mock",
    "runtime_profile": "gpu_single",
    "task_config": "kb9_cuda",
    "output_root": "runs",
}

_DEFAULT_PATHS = {
    "kernelbench_root": "/path/to/KernelBench",
    "default_manifest": "configs/tasks/kb9_cuda.yaml",
    "paper_manifest": "configs/tasks/kb_full_250.yaml",
    "env_file": ".env",
}

_PROFILE_DIRS = {
    "experiments": _CONFIG_ROOT / "experiments",
    "tasks": _CONFIG_ROOT / "tasks",
    "providers": _CONFIG_ROOT / "models" / "providers",
    "routes": _CONFIG_ROOT / "models" / "routes",
    "search": _CONFIG_ROOT / "search",
    "evaluators": _CONFIG_ROOT / "evaluation" / "evaluators",
    "measurement": _CONFIG_ROOT / "evaluation" / "measurement",
    "runtime": _CONFIG_ROOT / "runtime",
}

_LEGACY_PRESET_ALIASES = {
    "default": "quick_local",
    "smoke": "quick_local",
    "paper-pilot": "paper_mini",
}

_LEGACY_EVALUATION_PROFILE_ALIASES = {
    "kernelbench_reduced_v1": "quick",
    "kernelbench_profile_paper_mini_v1": "mini",
    "kernelbench_paper_mini_v1": "mini",
    "kernelbench_profile_paper_full_v1": "full",
    "kernelbench_paper_full_v1": "full",
    "kernelbench_cuda_reduced_v1": "quick",
}

_LEGACY_EVALUATOR_ALIASES = {
    "local": "local",
    "paper": "paper",
}

_CANONICAL_LEGACY_PRESET = {
    "quick_local": "default",
    "paper_like": "paper-pilot",
    "aggressive": "paper-pilot",
}

_CANONICAL_LEGACY_MEASUREMENT = {
    "quick": "kernelbench_reduced_v1",
    "mini": "kernelbench_profile_paper_mini_v1",
    "full": "kernelbench_profile_paper_full_v1",
}

_CANONICAL_LEGACY_EVALUATOR = {
    "local": "local",
    "paper": "paper",
}


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value
        loaded[key] = value
    return loaded


def workflow_choices(path: str | Path | None = None) -> list[str]:
    del path
    return list(_WORKFLOWS)


def backend_choices(path: str | Path | None = None) -> list[str]:
    del path
    return list(_BACKENDS)


def experiment_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("experiments")


def run_profile_choices(path: str | Path | None = None) -> list[str]:
    return experiment_choices(path)


def task_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("tasks")


def search_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("search")


def evaluator_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("evaluators")


def measurement_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("measurement")


def agent_provider_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("routes")


def runtime_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("runtime")


def kernelbench_preset_choices(path: str | Path | None = None) -> list[str]:
    del path
    return list(_LEGACY_PRESET_ALIASES.keys())


def kernelbench_evaluation_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return list(_LEGACY_EVALUATION_PROFILE_ALIASES.keys())


def kernelbench_evaluator_choices(path: str | Path | None = None) -> list[str]:
    del path
    return list(_LEGACY_EVALUATOR_ALIASES.keys())


def provider_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("providers")


def default_setting(name: str, path: str | Path | None = None) -> Any:
    del path
    return _DEFAULTS[name]


def path_setting(name: str, path: str | Path | None = None) -> str:
    del path
    runtime_name = default_setting("runtime_profile")
    runtime_settings = runtime_profile(runtime_name)
    if name in runtime_settings:
        return str(runtime_settings[name])
    return str(_DEFAULT_PATHS[name])


def experiment_profile(name: str) -> dict[str, Any]:
    selected = _load_profile("experiments", name)
    if not selected:
        return _fallback_experiment(name)
    return copy.deepcopy(selected)


def run_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    return experiment_profile(name)


def task_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    return _load_profile("tasks", name)


def search_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    return _load_profile("search", name)


def evaluator_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    return _load_profile("evaluators", name)


def measurement_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    return _load_profile("measurement", name)


def agent_provider_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    return _load_profile("routes", name)


def runtime_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    selected = _load_profile("runtime", name)
    return copy.deepcopy(selected)


def provider_defaults(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    selected = _load_profile("providers", name)
    return copy.deepcopy(selected)


def resolve_run_profile(explicit: str | None, legacy_preset: str | None = None, path: str | Path | None = None) -> str:
    del path
    if explicit:
        return str(explicit)
    if legacy_preset:
        return _LEGACY_PRESET_ALIASES.get(str(legacy_preset), _DEFAULTS["run_profile"])
    return str(_DEFAULTS["run_profile"])


def resolve_search_profile(
    explicit: str | None,
    run_profile_name: str,
    legacy_preset: str | None = None,
    path: str | Path | None = None,
) -> str:
    del path
    if explicit:
        return str(explicit)
    if legacy_preset == "paper-pilot":
        return "paper_like"
    if legacy_preset in {"default", "smoke"}:
        return "quick_local"
    selected = run_profile(run_profile_name)
    return str(selected.get("search", "quick_local"))


def resolve_evaluator_profile(
    explicit: str | None,
    run_profile_name: str,
    legacy_evaluator: str | None = None,
    path: str | Path | None = None,
) -> str:
    del path
    if explicit:
        return str(explicit)
    if legacy_evaluator:
        return _LEGACY_EVALUATOR_ALIASES.get(str(legacy_evaluator), "local")
    selected = run_profile(run_profile_name)
    return str(selected.get("evaluator", "local"))


def resolve_measurement_profile(
    explicit: str | None,
    run_profile_name: str,
    legacy_evaluation_profile: str | None = None,
    path: str | Path | None = None,
) -> str:
    del path
    if explicit:
        return str(explicit)
    if legacy_evaluation_profile:
        return _LEGACY_EVALUATION_PROFILE_ALIASES.get(str(legacy_evaluation_profile), "quick")
    selected = run_profile(run_profile_name)
    return str(selected.get("measurement", "quick"))


def resolve_agent_provider_profile(
    explicit: str | None,
    run_profile_name: str | None = None,
    path: str | Path | None = None,
) -> str:
    del path
    if explicit:
        return str(explicit)
    if run_profile_name:
        selected = run_profile(run_profile_name)
        return str(selected.get("route", _DEFAULTS["agent_provider_profile"]))
    return str(_DEFAULTS["agent_provider_profile"])


def resolve_task_profile(
    explicit: str | None,
    run_profile_name: str | None = None,
) -> str:
    if explicit:
        return str(explicit)
    if run_profile_name:
        selected = run_profile(run_profile_name)
        return str(selected.get("tasks", _DEFAULTS["task_config"]))
    return str(_DEFAULTS["task_config"])


def resolve_runtime_profile(
    explicit: str | None,
    run_profile_name: str | None = None,
) -> str:
    if explicit:
        return str(explicit)
    if run_profile_name:
        selected = run_profile(run_profile_name)
        return str(selected.get("runtime", _DEFAULTS["runtime_profile"]))
    return str(_DEFAULTS["runtime_profile"])


def legacy_preset_name(search_profile_name: str | None) -> str:
    return _CANONICAL_LEGACY_PRESET.get(str(search_profile_name or ""), "default")


def legacy_evaluation_profile_name(measurement_profile_name: str | None) -> str:
    return _CANONICAL_LEGACY_MEASUREMENT.get(str(measurement_profile_name or ""), "kernelbench_reduced_v1")


def legacy_kernelbench_evaluator_name(evaluator_profile_name: str | None) -> str:
    return _CANONICAL_LEGACY_EVALUATOR.get(str(evaluator_profile_name or ""), "local")


def kernelbench_evaluator_kind(evaluator_profile_name: str | None, path: str | Path | None = None) -> str:
    del path
    settings = evaluator_profile(str(evaluator_profile_name or "local"))
    return str(settings.get("kernelbench_evaluator", "local"))


def profile_path(group: str, name: str) -> Path:
    return _resolve_profile_path(group, name)


def _profile_names(group: str) -> list[str]:
    directory = _PROFILE_DIRS[group]
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def _load_profile(group: str, name: str) -> dict[str, Any]:
    path = _resolve_profile_path(group, name)
    if not path.exists():
        return {}
    payload = _load_yaml(path)
    base_name = payload.pop("base", None)
    if base_name:
        base_payload = _load_profile(group, str(base_name))
        return _deep_merge(base_payload, payload)
    return copy.deepcopy(payload)


def _resolve_profile_path(group: str, name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    directory = _PROFILE_DIRS[group]
    if candidate.suffix in {".yaml", ".yml"}:
        return directory / candidate.name
    return directory / f"{name}.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _fallback_experiment(name: str) -> dict[str, Any]:
    if name == "paper_full":
        return {
            "name": "paper_full",
            "description": "Full paper-style batch run.",
            "workflow": "stark",
            "backend": "triton",
            "provider": "openai-compatible",
            "route": "all_gpt",
            "tasks": "kb_full_250",
            "search": "paper_like",
            "evaluator": "paper",
            "measurement": "full",
            "runtime": "gpu_single",
            "output_root": "runs",
        }
    if name == "paper_mini":
        return {
            "name": "paper_mini",
            "description": "Default 9-task CUDA baseline with GPT plan/debug and cudaLLM code/search.",
            "workflow": "stark",
            "backend": "cuda",
            "provider": "openai-compatible",
            "route": "codeagent_cudallm",
            "tasks": "kb9_cuda",
            "search": "baseline",
            "evaluator": "paper",
            "measurement": "mini",
            "runtime": "gpu_single",
            "output_root": "runs",
        }
    return {
        "name": "quick_local",
        "description": "Fast local smoke run.",
        "workflow": "stark",
        "backend": "triton",
        "provider": "mock",
        "route": "all_mock",
        "tasks": "kb_smoke",
        "search": "quick_local",
        "evaluator": "local",
        "measurement": "quick",
        "runtime": "gpu_single",
        "output_root": "runs",
    }
