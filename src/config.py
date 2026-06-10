"""Layered YAML config loader for the KernelWeaver baseline.

The public mental model is intentionally simple:
- Users choose `tasks`, `backend`, `route`, and `profile`.
- `profile` only expands into `search`, `evaluator`, and `measurement`.

Some lightweight legacy aliases are still accepted when loading older runs,
but the recommended names are now only `quick`, `paper`, and `main`.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .backends import KERNELBENCH_BACKENDS


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _REPO_ROOT / "configs"

_WORKFLOWS = ["stark", "sampling", "reflexion", "search-agent", "ma-only"]
_BACKENDS = list(KERNELBENCH_BACKENDS)

_DEFAULTS: dict[str, Any] = {
    "workflow": "stark",
    "backend": "cuda",
    "provider": "openai-compatible",
    "run_profile": "quick",
    "agent_provider_profile": "codeagent_cudallm",
    "runtime_profile": "gpu_single",
    "task_config": "main_l1_15",
    "deliberation_profile": "quick",
    "output_root": "runs",
}

_DEFAULT_PATHS = {
    "kernelbench_root": "/path/to/KernelBench",
    "default_manifest": "configs/tasks/main_l1_15.yaml",
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
    "deliberation": _CONFIG_ROOT / "deliberation",
    "runtime": _CONFIG_ROOT / "runtime",
}

_LEGACY_EXPERIMENT_ALIASES = {
    "default": "quick",
    "smoke": "quick",
    "quick_local": "quick",
    "paper-pilot": "paper",
    "paper_mini": "main",
    "paper_full": "paper",
}

_LEGACY_EVALUATION_PROFILE_ALIASES = {
    "kernelbench_reduced_v1": "quick",
    "kernelbench_profile_paper_mini_v1": "main",
    "kernelbench_paper_mini_v1": "main",
    "kernelbench_profile_paper_full_v1": "paper",
    "kernelbench_paper_full_v1": "paper",
    "kernelbench_cuda_reduced_v1": "quick",
}

_LEGACY_EVALUATOR_ALIASES = {
    "local": "quick",
    "paper": "paper",
}

_CANONICAL_LEGACY_PRESET = {
    "quick": "default",
    "paper": "paper-pilot",
    "main": "paper_mini",
}

_CANONICAL_LEGACY_MEASUREMENT = {
    "quick": "kernelbench_reduced_v1",
    "paper": "kernelbench_profile_paper_full_v1",
    "main": "kernelbench_profile_paper_mini_v1",
}

_CANONICAL_LEGACY_EVALUATOR = {
    "quick": "paper",
    "paper": "paper",
    "main": "paper",
}

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


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
        if key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ[key]
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


def deliberation_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return _profile_names("deliberation")


def kernelbench_preset_choices(path: str | Path | None = None) -> list[str]:
    del path
    return sorted(_LEGACY_EXPERIMENT_ALIASES.keys())


def kernelbench_evaluation_profile_choices(path: str | Path | None = None) -> list[str]:
    del path
    return sorted(_LEGACY_EVALUATION_PROFILE_ALIASES.keys())


def kernelbench_evaluator_choices(path: str | Path | None = None) -> list[str]:
    del path
    return sorted(_LEGACY_EVALUATOR_ALIASES.keys())


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
    resolved = _LEGACY_EXPERIMENT_ALIASES.get(str(name), str(name))
    selected = _load_profile("experiments", resolved)
    if not selected:
        return _fallback_experiment(resolved)
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


def deliberation_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    selected = _load_profile("deliberation", name)
    return copy.deepcopy(selected)


def provider_defaults(name: str, path: str | Path | None = None) -> dict[str, Any]:
    del path
    selected = _load_profile("providers", name)
    return copy.deepcopy(selected)


def resolve_run_profile(explicit: str | None, legacy_preset: str | None = None, path: str | Path | None = None) -> str:
    del path
    if explicit:
        return _LEGACY_EXPERIMENT_ALIASES.get(str(explicit), str(explicit))
    if legacy_preset:
        return _LEGACY_EXPERIMENT_ALIASES.get(str(legacy_preset), _DEFAULTS["run_profile"])
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
    if legacy_preset:
        legacy_run = _LEGACY_EXPERIMENT_ALIASES.get(str(legacy_preset))
        if legacy_run:
            return str(run_profile(legacy_run).get("search", "quick"))
    selected = run_profile(run_profile_name)
    return str(selected.get("search", "quick"))


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
        mapped = _LEGACY_EVALUATOR_ALIASES.get(str(legacy_evaluator))
        if mapped:
            return str(mapped)
    selected = run_profile(run_profile_name)
    return str(selected.get("evaluator", "quick"))


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
        mapped = _LEGACY_EVALUATION_PROFILE_ALIASES.get(str(legacy_evaluation_profile))
        if mapped:
            return str(mapped)
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


def resolve_task_profile(explicit: str | None, run_profile_name: str | None = None) -> str:
    if explicit:
        return str(explicit)
    if run_profile_name:
        selected = run_profile(run_profile_name)
        return str(selected.get("tasks", _DEFAULTS["task_config"]))
    return str(_DEFAULTS["task_config"])


def resolve_runtime_profile(explicit: str | None, run_profile_name: str | None = None) -> str:
    if explicit:
        return str(explicit)
    if run_profile_name:
        selected = run_profile(run_profile_name)
        return str(selected.get("runtime", _DEFAULTS["runtime_profile"]))
    return str(_DEFAULTS["runtime_profile"])


def resolve_deliberation_profile(explicit: str | None, run_profile_name: str | None = None) -> str:
    if explicit:
        return str(explicit)
    if run_profile_name:
        selected = run_profile(run_profile_name)
        return str(selected.get("deliberation", _DEFAULTS["deliberation_profile"]))
    return str(_DEFAULTS["deliberation_profile"])


def legacy_preset_name(search_profile_name: str | None) -> str:
    return _CANONICAL_LEGACY_PRESET.get(str(search_profile_name or ""), "default")


def legacy_evaluation_profile_name(measurement_profile_name: str | None) -> str:
    return _CANONICAL_LEGACY_MEASUREMENT.get(str(measurement_profile_name or ""), "kernelbench_reduced_v1")


def legacy_kernelbench_evaluator_name(evaluator_profile_name: str | None) -> str:
    return _CANONICAL_LEGACY_EVALUATOR.get(str(evaluator_profile_name or ""), "paper")


def kernelbench_evaluator_kind(evaluator_profile_name: str | None, path: str | Path | None = None) -> str:
    del path
    settings = evaluator_profile(str(evaluator_profile_name or "quick"))
    return str(settings.get("kernelbench_evaluator", "paper"))


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
    return _expand_env_placeholders(payload)


def _expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            default_value = match.group(2)
            if env_name in os.environ:
                return os.environ[env_name]
            if default_value is not None:
                return default_value
            return match.group(0)

        return _ENV_PATTERN.sub(_replace, value)
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _fallback_experiment(name: str) -> dict[str, Any]:
    if name == "paper":
        return {
            "name": "paper",
            "description": "Paper-style STARK configuration.",
            "workflow": "stark",
            "backend": "cuda",
            "provider": "openai-compatible",
            "route": "all_gpt",
            "tasks": "kb_full_250",
            "search": "paper",
            "evaluator": "paper",
            "measurement": "paper",
            "runtime": "gpu_single",
            "semantics": {"enabled": True, "mode": "rule", "max_anchor_hints": 6},
            "diagnostics": {"enabled": False, "mode": "disabled", "timeout_seconds": 300, "warmup_runs": 2, "profile_runs": 3},
            "output_root": "runs",
        }
    if name == "main":
        return {
            "name": "main",
            "description": "Default formal KernelWeaver configuration.",
            "workflow": "stark",
            "backend": "cuda",
            "provider": "openai-compatible",
            "route": "codeagent_cudallm",
            "tasks": "main_l1_15",
            "search": "main",
            "evaluator": "main",
            "measurement": "main",
            "runtime": "gpu_single",
            "semantics": {"enabled": True, "mode": "rule", "max_anchor_hints": 6},
            "diagnostics": {"enabled": True, "mode": "machine_check_v1", "timeout_seconds": 300, "warmup_runs": 2, "profile_runs": 3},
            "output_root": "runs",
        }
    return {
        "name": "quick",
        "description": "Fast smoke configuration.",
        "workflow": "stark",
        "backend": "triton",
        "provider": "mock",
        "route": "all_mock",
        "tasks": "kb_smoke",
        "search": "quick",
        "evaluator": "quick",
        "measurement": "quick",
        "runtime": "gpu_single",
        "semantics": {"enabled": True, "mode": "rule", "max_anchor_hints": 6},
        "diagnostics": {"enabled": False, "mode": "disabled", "timeout_seconds": 120, "warmup_runs": 1, "profile_runs": 1},
        "output_root": "runs",
    }
