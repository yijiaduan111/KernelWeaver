"""Command line entrypoints for the KernelWeaver baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    agent_provider_profile,
    agent_provider_profile_choices,
    backend_choices,
    default_setting,
    deliberation_profile,
    deliberation_profile_choices,
    evaluator_profile,
    evaluator_profile_choices,
    experiment_profile,
    kernelbench_evaluator_kind,
    legacy_evaluation_profile_name,
    legacy_preset_name,
    load_env_file,
    measurement_profile,
    measurement_profile_choices,
    path_setting,
    profile_path,
    provider_choices,
    provider_defaults,
    resolve_agent_provider_profile,
    resolve_deliberation_profile,
    resolve_evaluator_profile,
    resolve_measurement_profile,
    resolve_run_profile,
    resolve_runtime_profile,
    resolve_search_profile,
    resolve_task_profile,
    run_profile_choices,
    runtime_profile,
    runtime_profile_choices,
    search_profile,
    search_profile_choices,
    task_profile_choices,
    workflow_choices,
)
from .core.loader import KernelBenchLoader
from .diagnostics import build_task_diagnostics
from .deliberation.runner import MultiModelDeliberationRunner
from .core.workflow import bootstrap_stark_root, run_stark
from .demo import build_demo_tasks
from .evaluation import (
    DemoEvaluator,
    KernelBenchPaperEvaluator,
    IsolatedEvaluator,
    TritonEvaluator,
    load_validation,
    verify_kernelbench_run,
)
from .experiment import (
    aggregate_batch_rows,
    batch_output_dir_name,
    candidate_attempt_stats,
    format_speedup,
    load_task_manifest,
    runtime_for_mode,
    speedup_for_mode,
    write_batch_csv,
    build_paper_summary_report,
    write_paper_summary_report,
)
from .io import load_run, save_run
from .models import StarkConfig
from .providers import ClaudeCompatibleConfig, ClaudeCompatibleProvider, GeminiCompatibleConfig, GeminiCompatibleProvider, LocalCudaLLMConfig, LocalCudaLLMProvider, MockProvider, OpenAICompatibleProvider, RoleRoutedProvider
from .triton_tasks import build_triton_tasks
from .utils import shorten_runtime

DEFAULT_KERNELBENCH_ROOT = path_setting("kernelbench_root")
DEFAULT_KERNELBENCH_MANIFEST = path_setting("default_manifest")
DEFAULT_KERNELBENCH_PAPER_MANIFEST = path_setting("paper_manifest")
WORKFLOW_CHOICES = workflow_choices()
BACKEND_CHOICES = backend_choices()
EXPERIMENT_CHOICES = run_profile_choices()
SEARCH_CHOICES = search_profile_choices()
EVALUATOR_CHOICES = evaluator_profile_choices()
MEASUREMENT_CHOICES = measurement_profile_choices()
DELIBERATION_CHOICES = deliberation_profile_choices()
PROVIDER_CHOICES = provider_choices()
ROUTE_CHOICES = agent_provider_profile_choices()
TASK_CHOICES = task_profile_choices()
RUNTIME_CHOICES = runtime_profile_choices()
ROLE_PROVIDER_CHOICES = ["inherit", *PROVIDER_CHOICES]


def _demo_task_map() -> dict[str, Any]:
    return {task.name: task for task in build_demo_tasks()}


def _triton_task_map() -> dict[str, Any]:
    return {task.name: task for task in build_triton_tasks()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stark", description="Run the KernelWeaver baseline workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_demo = subparsers.add_parser("run-demo", help="Run a demo task.")
    _add_shared_run_arguments(run_demo, sorted(_demo_task_map().keys()))

    run_triton = subparsers.add_parser("run-triton", help="Run a built-in Triton toy task.")
    _add_shared_run_arguments(run_triton, None)

    run_kernelbench = subparsers.add_parser("run-kernelbench", help="Run one KernelBench task.")
    _add_kernelbench_run_arguments(run_kernelbench)

    run_kernelbench_batch = subparsers.add_parser("run-kernelbench-batch", help="Run a task manifest.")
    run_kernelbench_batch.add_argument("--manifest", default=None, help="Path to a YAML or JSON manifest file.")
    _add_kernelbench_run_arguments(run_kernelbench_batch, include_problem_flags=False)

    verify_kernelbench = subparsers.add_parser("verify-kernelbench", help="Re-run the best KernelBench candidate.")
    verify_kernelbench.add_argument("path", type=str, help="Path to a saved run.json")
    verify_kernelbench.add_argument("--kernelbench-root", default=DEFAULT_KERNELBENCH_ROOT)
    verify_kernelbench.add_argument("--output", type=str, default=None)

    report_paper = subparsers.add_parser("report-paper", help="Build a paper-style summary report.")
    report_paper.add_argument("paths", nargs="+", type=str, help="Summary files or directories that contain summary.json")
    report_paper.add_argument("--output-dir", required=True, type=str)
    report_paper.add_argument("--title", default="KernelWeaver Paper Summary")

    show_parser = subparsers.add_parser("show-run", help="Show a saved run summary.")
    show_parser.add_argument("path", type=str)

    summarize = subparsers.add_parser("summarize-runs", help="Print one compact line per saved run.")
    summarize.add_argument("paths", nargs="+", type=str)
    return parser


def _add_shared_run_arguments(parser: argparse.ArgumentParser, task_choices: list[str] | None) -> None:
    if task_choices is None:
        parser.add_argument("--task", required=True)
    else:
        parser.add_argument("--task", required=True, choices=task_choices)
    parser.add_argument("--experiment", "--run-profile", dest="run_profile", default=default_setting("run_profile"), choices=EXPERIMENT_CHOICES)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--provider", default=None, choices=PROVIDER_CHOICES)
    parser.add_argument("--search-config", "--search-profile", dest="search_profile", default=None, choices=SEARCH_CHOICES)
    parser.add_argument("--route-config", "--agent-provider-profile", dest="agent_provider_profile", default=None, choices=ROUTE_CHOICES)
    parser.add_argument("--deliberation-config", "--deliberation-profile", dest="deliberation_profile", default=None, choices=DELIBERATION_CHOICES)
    _add_provider_routing_arguments(parser)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--verbose", action="store_true")


def _add_kernelbench_run_arguments(parser: argparse.ArgumentParser, include_problem_flags: bool = True) -> None:
    parser.add_argument("--experiment", "--run-profile", dest="run_profile", default="main", choices=EXPERIMENT_CHOICES)
    parser.add_argument("--kernelbench-root", default=None)
    if include_problem_flags:
        parser.add_argument("--level", type=int, required=True)
        parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--backend", default=None, choices=BACKEND_CHOICES)
    parser.add_argument("--workflow", default=None, choices=WORKFLOW_CHOICES)
    parser.add_argument("--task-config", default=None, choices=TASK_CHOICES, help="Advanced: choose a named task manifest under configs/tasks.")
    parser.add_argument("--runtime-config", default=None, choices=RUNTIME_CHOICES, help="Advanced: choose a runtime profile under configs/runtime.")
    parser.add_argument("--search-config", "--search-profile", dest="search_profile", default=None, choices=SEARCH_CHOICES)
    parser.add_argument("--evaluator-config", "--evaluator-profile", dest="evaluator_profile", default=None, choices=EVALUATOR_CHOICES)
    parser.add_argument("--measurement-config", "--measurement-profile", dest="measurement_profile", default=None, choices=MEASUREMENT_CHOICES)
    parser.add_argument("--deliberation-config", "--deliberation-profile", dest="deliberation_profile", default=None, choices=DELIBERATION_CHOICES)
    parser.add_argument("--route-config", "--agent-provider-profile", dest="agent_provider_profile", default=None, choices=ROUTE_CHOICES)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--provider", default=None, choices=PROVIDER_CHOICES)
    _add_provider_routing_arguments(parser)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--verbose", action="store_true")


def _add_provider_routing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-provider", default=None, choices=ROLE_PROVIDER_CHOICES)
    parser.add_argument("--code-provider", default=None, choices=ROLE_PROVIDER_CHOICES)
    parser.add_argument("--debug-provider", default=None, choices=ROLE_PROVIDER_CHOICES)
    parser.add_argument("--search-provider", default=None, choices=ROLE_PROVIDER_CHOICES)


def _resolve_run_name(args: argparse.Namespace) -> str:
    return resolve_run_profile(getattr(args, "run_profile", None))


def _resolve_search_name(args: argparse.Namespace, run_name: str) -> str:
    return resolve_search_profile(getattr(args, "search_profile", None), run_name)


def _resolve_evaluator_name(args: argparse.Namespace, run_name: str) -> str:
    return resolve_evaluator_profile(getattr(args, "evaluator_profile", None), run_name)


def _resolve_measurement_name(args: argparse.Namespace, run_name: str) -> str:
    return resolve_measurement_profile(getattr(args, "measurement_profile", None), run_name)


def _resolve_route_name(args: argparse.Namespace, run_name: str) -> str:
    return resolve_agent_provider_profile(getattr(args, "agent_provider_profile", None), run_name)


def _resolve_deliberation_name(args: argparse.Namespace, run_name: str) -> str:
    return resolve_deliberation_profile(getattr(args, "deliberation_profile", None), run_name)


def _resolve_task_name(args: argparse.Namespace, run_name: str) -> str:
    return resolve_task_profile(getattr(args, "task_config", None), run_name)


def _resolve_runtime_name(args: argparse.Namespace, run_name: str) -> str:
    return resolve_runtime_profile(getattr(args, "runtime_config", None), run_name)


def _resolve_runtime_settings(args: argparse.Namespace, run_name: str) -> dict[str, Any]:
    return runtime_profile(_resolve_runtime_name(args, run_name))


def _resolve_provider_name(args: argparse.Namespace, run_name: str) -> str:
    if getattr(args, "provider", None):
        return str(args.provider)
    experiment = experiment_profile(run_name)
    return str(experiment.get("provider", default_setting("provider")))


def _resolve_backend(args: argparse.Namespace, run_name: str) -> str:
    if getattr(args, "backend", None):
        return str(args.backend)
    experiment = experiment_profile(run_name)
    return str(experiment.get("backend", default_setting("backend")))


def _resolve_workflow(args: argparse.Namespace, run_name: str) -> str:
    if getattr(args, "workflow", None):
        return str(args.workflow)
    experiment = experiment_profile(run_name)
    return str(experiment.get("workflow", default_setting("workflow")))


def _resolve_semantics_settings(run_name: str) -> dict[str, Any]:
    experiment = experiment_profile(run_name)
    raw = experiment.get("semantics", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "mode": str(raw.get("mode", "rule")),
        "max_anchor_hints": int(raw.get("max_anchor_hints", 6)),
    }



def _resolve_diagnostics_settings(run_name: str) -> dict[str, Any]:
    experiment = experiment_profile(run_name)
    raw = experiment.get("diagnostics", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "mode": str(raw.get("mode", "root_ncu_v1")),
        "timeout_seconds": int(raw.get("timeout_seconds", 300)),
        "warmup_runs": int(raw.get("warmup_runs", 2)),
        "profile_runs": int(raw.get("profile_runs", 3)),
    }


def _resolve_env_file(args: argparse.Namespace, run_name: str) -> str:
    if getattr(args, "env_file", None):
        return str(args.env_file)
    runtime_settings = _resolve_runtime_settings(args, run_name)
    return str(runtime_settings.get("env_file", path_setting("env_file")))


def _resolve_kernelbench_root(args: argparse.Namespace, run_name: str) -> str:
    if getattr(args, "kernelbench_root", None):
        return str(args.kernelbench_root)
    runtime_settings = _resolve_runtime_settings(args, run_name)
    return str(runtime_settings.get("kernelbench_root", path_setting("kernelbench_root")))


def _resolve_manifest_path(args: argparse.Namespace, run_name: str) -> Path:
    if getattr(args, "manifest", None):
        return Path(str(args.manifest))
    return profile_path("tasks", _resolve_task_name(args, run_name))


def _resolve_provider_routing(args: argparse.Namespace, run_name: str) -> dict[str, str]:
    provider_name = _resolve_provider_name(args, run_name)
    route_name = _resolve_route_name(args, run_name)
    route = agent_provider_profile(route_name)

    def _pick(explicit_attr: str, route_key: str) -> str:
        explicit = getattr(args, explicit_attr, None)
        raw_value = explicit if explicit is not None else route.get(route_key, "inherit")
        chosen = str(raw_value or "inherit")
        return provider_name if chosen == "inherit" else chosen

    return {
        "provider_name": provider_name,
        "agent_provider_profile": route_name,
        "plan_provider": _pick("plan_provider", "plan_provider"),
        "code_provider": _pick("code_provider", "code_provider"),
        "debug_provider": _pick("debug_provider", "debug_provider"),
        "search_provider": _pick("search_provider", "search_provider"),
    }


def _provider_overrides(args: argparse.Namespace, run_name: str) -> dict[str, Any]:
    search_settings = search_profile(_resolve_search_name(args, run_name))
    return {
        key: search_settings[key]
        for key in ["plan_temperature", "code_temperature", "debug_temperature", "timeout_seconds", "max_retries"]
        if key in search_settings
    }


def _config_payload(raw: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key in allowed_keys}


def _instantiate_single_provider(name: str, overrides: dict[str, Any] | None = None):
    overrides = overrides or {}
    if name == "mock":
        return MockProvider()
    if name == "openai-compatible":
        provider = OpenAICompatibleProvider.from_env(provider_defaults(name))
        return provider.with_overrides(**overrides) if overrides else provider
    if name == "claude-compatible":
        provider = ClaudeCompatibleProvider.from_env(
            _config_payload(
                provider_defaults(name),
                set(ClaudeCompatibleConfig.__dataclass_fields__.keys()),
            )
        )
        return provider.with_overrides(**overrides) if overrides else provider
    if name == "gemini-compatible":
        provider = GeminiCompatibleProvider.from_env(
            _config_payload(
                provider_defaults(name),
                set(GeminiCompatibleConfig.__dataclass_fields__.keys()),
            )
        )
        return provider.with_overrides(**overrides) if overrides else provider
    if name == "local-cudallm":
        provider = LocalCudaLLMProvider(
            LocalCudaLLMConfig(
                **_config_payload(
                    provider_defaults(name),
                    set(LocalCudaLLMConfig.__dataclass_fields__.keys()),
                )
            )
        )
        return provider.with_overrides(**overrides) if overrides else provider
    raise SystemExit(f"Unsupported provider: {name}")


def _prepare_runtime_and_env(args: argparse.Namespace, run_name: str) -> None:
    runtime_settings = _resolve_runtime_settings(args, run_name)
    if runtime_settings.get("cuda_visible_devices") and "CUDA_VISIBLE_DEVICES" not in __import__("os").environ:
        __import__("os").environ["CUDA_VISIBLE_DEVICES"] = str(runtime_settings["cuda_visible_devices"])
    load_env_file(_resolve_env_file(args, run_name))


def _build_provider(args: argparse.Namespace, run_name: str):
    routing = _resolve_provider_routing(args, run_name)
    _prepare_runtime_and_env(args, run_name)
    overrides = _provider_overrides(args, run_name)
    resolved_names = {routing["plan_provider"], routing["code_provider"], routing["debug_provider"], routing["search_provider"]}
    instances = {name: _instantiate_single_provider(name, overrides) for name in sorted(resolved_names)}
    if len(instances) == 1:
        return instances[next(iter(instances))]
    return RoleRoutedProvider(
        plan_provider=instances[routing["plan_provider"]],
        code_provider=instances[routing["code_provider"]],
        debug_provider=instances[routing["debug_provider"]],
        search_provider=instances[routing["search_provider"]],
    )


def _build_deliberation_runner(args: argparse.Namespace, run_name: str, config: StarkConfig) -> MultiModelDeliberationRunner | None:
    if not config.deliberation_enabled:
        return None
    _prepare_runtime_and_env(args, run_name)
    timeout_seconds = _provider_overrides(args, run_name).get("timeout_seconds", 300)
    providers = {}
    for name in config.deliberation_providers:
        try:
            providers[name] = _instantiate_single_provider(name, {"timeout_seconds": timeout_seconds})
        except ValueError as exc:
            print(
                f"[KernelWeaver] Skipping deliberation provider {name}: {exc}",
                file=sys.stderr,
            )
    if not providers:
        print(
            "[KernelWeaver] No deliberation providers are available; falling back to the base workflow.",
            file=sys.stderr,
        )
        return None
    return MultiModelDeliberationRunner(
        providers=providers,
        max_strategies=config.deliberation_max_strategies,
        strategies_per_model=config.deliberation_strategies_per_model,
        proposal_temperature=config.deliberation_proposal_temperature,
        review_temperature=config.deliberation_review_temperature,
        mode=config.deliberation_mode,
    )


def _apply_deliberation(task, config: StarkConfig, runner: MultiModelDeliberationRunner | None) -> None:
    if runner is None or not config.deliberation_enabled:
        return
    if config.verbose:
        print(
            f"[deliberation] start mode={config.deliberation_mode} providers={','.join(config.deliberation_providers)}",
            flush=True,
        )
    started = time.time()
    task.strategy_portfolio = runner.run(task, config)
    if config.verbose:
        for event in runner.last_events:
            elapsed = "" if event.elapsed_seconds is None else f" elapsed={event.elapsed_seconds:.3f}s"
            detail = f" {event.detail}" if event.detail else ""
            print(
                f"[deliberation] phase={event.phase} provider={event.provider_name} status={event.status}{elapsed}{detail}",
                flush=True,
            )
        strategy_count = len(task.strategy_portfolio.strategies) if task.strategy_portfolio else 0
        print(
            f"[deliberation] done elapsed={time.time() - started:.3f}s strategies={strategy_count}",
            flush=True,
        )


def _apply_diagnostics(task, config: StarkConfig, *, candidate_code: str | None = None, root_evaluation=None) -> None:
    task.diagnostics_profile = build_task_diagnostics(
        task,
        config,
        candidate_code=candidate_code,
        root_evaluation=root_evaluation,
    )
    if config.verbose and task.diagnostics_profile is not None:
        ncu_profile = task.diagnostics_profile.ncu_profile
        raw_metric_count = len(getattr(ncu_profile, "raw_metrics", {}) or {}) if ncu_profile is not None else 0
        kernel_name = getattr(ncu_profile, "kernel_name", None) if ncu_profile is not None else None
        print(
            f"[diagnostics] enabled={task.diagnostics_profile.enabled} mode={task.diagnostics_profile.mode} "
            f"kernel={kernel_name or 'none'} raw_metrics={raw_metric_count}",
            flush=True,
        )


def _prepare_kernelbench_root_state(task, config: StarkConfig, evaluator, deliberation_runner: MultiModelDeliberationRunner | None):
    needs_root_guidance = bool(getattr(config, "diagnostics_enabled", False) or deliberation_runner is not None)
    if not needs_root_guidance:
        return None
    initial_state = bootstrap_stark_root(task, config, evaluator)
    tree, root_eval, _stats, _debug_stats = initial_state
    _apply_diagnostics(
        task,
        config,
        candidate_code=tree.get_node(tree.root_id).code,
        root_evaluation=root_eval,
    )
    _apply_deliberation(task, config, deliberation_runner)
    return initial_state


def _close_provider(provider) -> None:
    close_fn = getattr(provider, "close", None)
    if callable(close_fn):
        close_fn()


def _build_config(args: argparse.Namespace, run_name: str) -> StarkConfig:
    search_name = _resolve_search_name(args, run_name)
    evaluator_name = _resolve_evaluator_name(args, run_name) if getattr(args, "command", "") in {"run-kernelbench", "run-kernelbench-batch"} else "quick"
    measurement_name = _resolve_measurement_name(args, run_name) if getattr(args, "command", "") in {"run-kernelbench", "run-kernelbench-batch"} else "quick"
    route = _resolve_provider_routing(args, run_name)
    search_settings = search_profile(search_name)
    evaluator_settings = evaluator_profile(evaluator_name)
    measure_settings = measurement_profile(measurement_name)
    semantics_settings = _resolve_semantics_settings(run_name)
    diagnostics_settings = _resolve_diagnostics_settings(run_name)
    deliberation_name = _resolve_deliberation_name(args, run_name)
    deliberation_settings = deliberation_profile(deliberation_name)
    max_attempts = int(getattr(args, "max_attempts", None) or search_settings.get("max_attempts", 6))
    epsilon = float(getattr(args, "epsilon", None) if getattr(args, "epsilon", None) is not None else search_settings.get("epsilon", 0.4))
    return StarkConfig(
        max_attempts=max_attempts,
        epsilon=epsilon,
        root_child_limit=int(search_settings.get("root_child_limit", 2)),
        dead_branch_threshold=int(search_settings.get("dead_branch_threshold", 2)),
        leaderboard_size=int(search_settings.get("leaderboard_size", 3)),
        context_limit=int(search_settings.get("context_limit", 5)),
        benchmark_loops=int(measure_settings.get("benchmark_loops", 50)),
        warmup_loops=int(measure_settings.get("warmup_loops", 5)),
        verbose=bool(getattr(args, "verbose", False)),
        run_profile=run_name,
        search_profile=search_name,
        evaluator_profile=evaluator_name,
        measurement_profile=measurement_name,
        provider_name=route["provider_name"],
        agent_provider_profile=route["agent_provider_profile"],
        plan_provider=route["plan_provider"],
        code_provider=route["code_provider"],
        debug_provider=route["debug_provider"],
        search_provider=route["search_provider"],
        preset=legacy_preset_name(search_name),
        evaluation_profile=legacy_evaluation_profile_name(measurement_name),
        kernelbench_evaluator=kernelbench_evaluator_kind(evaluator_name),
        num_correct_trials=int(measure_settings.get("num_correct_trials", 1)),
        num_perf_trials=int(measure_settings.get("num_perf_trials", 10)),
        paper_num_warmup=int(measure_settings.get("paper_num_warmup", 5)),
        paper_discard_first=int(measure_settings.get("paper_discard_first", 1)),
        timing_method=str(measure_settings.get("timing_method", "cuda_event")),
        reference_modes=list(evaluator_settings.get("reference_modes") or ["torch_eager"]),
        semantics_enabled=semantics_settings["enabled"],
        semantics_mode=semantics_settings["mode"],
        semantics_max_anchor_hints=semantics_settings["max_anchor_hints"],
        diagnostics_enabled=diagnostics_settings["enabled"],
        diagnostics_mode=diagnostics_settings["mode"],
        diagnostics_timeout_seconds=diagnostics_settings["timeout_seconds"],
        diagnostics_warmup_runs=diagnostics_settings["warmup_runs"],
        diagnostics_profile_runs=diagnostics_settings["profile_runs"],
        deliberation_enabled=bool(deliberation_settings.get("enabled", False)),
        deliberation_profile=deliberation_name,
        deliberation_mode=str(deliberation_settings.get("mode", "multi_model_v0")),
        deliberation_providers=list(deliberation_settings.get("providers") or []),
        deliberation_max_strategies=int(deliberation_settings.get("max_strategies", 10)),
        deliberation_strategies_per_model=int(deliberation_settings.get("strategies_per_model", 4)),
        deliberation_proposal_temperature=float(deliberation_settings.get("proposal_temperature", 0.4)),
        deliberation_review_temperature=float(deliberation_settings.get("review_temperature", 0.1)),
        evaluator_isolation=str(evaluator_settings.get("evaluator_isolation", "off")),
        evaluator_timeout_seconds=int(evaluator_settings.get("evaluator_timeout_seconds", 900)),
    )


def _kernelbench_evaluator(backend: str, evaluator_name: str, config: StarkConfig | None = None):
    del backend
    del evaluator_name
    evaluator = KernelBenchPaperEvaluator()
    if config is not None and config.evaluator_isolation == "candidate_subprocess":
        return IsolatedEvaluator(evaluator, timeout_seconds=config.evaluator_timeout_seconds)
    return evaluator


def _run_demo(args: argparse.Namespace) -> int:
    run_name = _resolve_run_name(args)
    task = _demo_task_map()[args.task]
    config = _build_config(args, run_name)
    deliberation_runner = _build_deliberation_runner(args, run_name, config)
    provider = _build_provider(args, run_name)
    try:
        _apply_diagnostics(task, config)
        _apply_deliberation(task, config, deliberation_runner)
        result = run_stark(task, config, provider, DemoEvaluator(), deliberation_runner=deliberation_runner)
        return _save_and_print(result, args.output_dir)
    finally:
        _close_provider(provider)
        if deliberation_runner is not None:
            deliberation_runner.close()


def _run_triton(args: argparse.Namespace) -> int:
    run_name = _resolve_run_name(args)
    task_map = _triton_task_map()
    if args.task not in task_map:
        raise SystemExit(f"Unknown Triton task: {args.task}")
    task = task_map[args.task]
    config = _build_config(args, run_name)
    deliberation_runner = _build_deliberation_runner(args, run_name, config)
    provider = _build_provider(args, run_name)
    try:
        _apply_diagnostics(task, config)
        _apply_deliberation(task, config, deliberation_runner)
        result = run_stark(task, config, provider, TritonEvaluator(), deliberation_runner=deliberation_runner)
        return _save_and_print(result, args.output_dir)
    finally:
        _close_provider(provider)
        if deliberation_runner is not None:
            deliberation_runner.close()


def _run_kernelbench(args: argparse.Namespace) -> int:
    run_name = _resolve_run_name(args)
    backend = _resolve_backend(args, run_name)
    workflow = _resolve_workflow(args, run_name)
    loader = KernelBenchLoader()
    config = _build_config(args, run_name)
    task = loader.load_official_problem(
        _resolve_kernelbench_root(args, run_name),
        args.level,
        args.problem_id,
        backend=backend,
        semantics_enabled=config.semantics_enabled,
        semantics_mode=config.semantics_mode,
        semantics_max_anchor_hints=config.semantics_max_anchor_hints,
    )
    deliberation_runner = _build_deliberation_runner(args, run_name, config)
    provider = _build_provider(args, run_name)
    evaluator = _kernelbench_evaluator(backend, config.evaluator_profile or "quick", config)
    try:
        initial_state = _prepare_kernelbench_root_state(task, config, evaluator, deliberation_runner)
        result = run_stark(
            task,
            config,
            provider,
            evaluator,
            deliberation_runner=deliberation_runner,
            initial_state=initial_state,
        )
        return _save_and_print(result, args.output_dir)
    finally:
        _close_provider(provider)
        if deliberation_runner is not None:
            deliberation_runner.close()


def _run_kernelbench_batch(args: argparse.Namespace) -> int:
    run_name = _resolve_run_name(args)
    backend = _resolve_backend(args, run_name)
    workflow = _resolve_workflow(args, run_name)
    kernelbench_root = _resolve_kernelbench_root(args, run_name)
    manifest = load_task_manifest(_resolve_manifest_path(args, run_name), kernelbench_root=kernelbench_root, default_backend=backend)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    loader = KernelBenchLoader()
    batch_config = _build_config(args, run_name)
    deliberation_runner = _build_deliberation_runner(args, run_name, batch_config)
    provider = _build_provider(args, run_name)
    rows: list[dict[str, Any]] = []
    try:
        for item in manifest["tasks"]:
            level = int(item["level"])
            problem_id = int(item["problem_id"])
            alias = str(item.get("alias") or f"L{level}_P{problem_id}")
            item_backend = str(item.get("backend") or backend)
            row = {
                "alias": alias,
                "level": level,
                "problem_id": problem_id,
                "backend": item_backend,
                "workflow": workflow,
                "run_profile": run_name,
                "search_profile": _resolve_search_name(args, run_name),
                "evaluator_profile": _resolve_evaluator_name(args, run_name),
                "measurement_profile": _resolve_measurement_name(args, run_name),
                "agent_provider_profile": _resolve_route_name(args, run_name),
                "status": "error",
                "error": None,
            }
            try:
                config = _build_config(args, run_name)
                task = loader.load_official_problem(
                    kernelbench_root,
                    level,
                    problem_id,
                    backend=item_backend,
                    semantics_enabled=config.semantics_enabled,
                    semantics_mode=config.semantics_mode,
                    semantics_max_anchor_hints=config.semantics_max_anchor_hints,
                )
                evaluator = _kernelbench_evaluator(item_backend, _resolve_evaluator_name(args, run_name), config)
                initial_state = _prepare_kernelbench_root_state(task, config, evaluator, deliberation_runner)
                task_output_dir = output_root / batch_output_dir_name(alias, level, problem_id)
                result = run_stark(
                    task,
                    config,
                    provider,
                    evaluator,
                    deliberation_runner=deliberation_runner,
                    initial_state=initial_state,
                )
                run_path = save_run(result, task_output_dir)
                validation_path = None
                best = result.nodes[result.best_node_id]
                validation = {}
                stats = candidate_attempt_stats(result)
                row.update(
                    {
                        "status": "ok",
                        "workflow": result.workflow,
                        "run_profile": result.run_profile,
                        "search_profile": result.search_profile,
                        "evaluator_profile": result.evaluator_profile,
                        "measurement_profile": result.measurement_profile,
                        "agent_provider_profile": result.config.agent_provider_profile,
                        "deliberation_profile": result.config.deliberation_profile,
                        "strategy_count": len(result.strategy_portfolio.strategies) if result.strategy_portfolio else 0,
                        "plan_provider": result.config.plan_provider,
                        "code_provider": result.config.code_provider,
                        "debug_provider": result.config.debug_provider,
                        "search_provider": result.config.search_provider,
                        "preset": result.preset,
                        "evaluation_profile": result.evaluation_profile,
                        "kernelbench_evaluator": result.kernelbench_evaluator,
                        "task_name": result.task_name,
                        "best_node_id": result.best_node_id,
                        "best_status": best.status,
                        "best_node_is_root": result.best_node_id == "root",
                        "best_correct": bool(best.compile_ok and best.correct),
                        "paper_fast1": bool(best.correct and isinstance(best.speedup, (int, float)) and best.speedup >= 1.0),
                        "run_path": str(run_path),
                        "validation_path": None,
                        "root_correct": bool(result.nodes.get("root") and result.nodes["root"].correct),
                        "non_root_correct": any(node.correct and node.node_id != "root" for node in result.nodes.values()),
                        "improved_over_reference": bool(best.runtime is not None and best.reference_runtime is not None and best.runtime < best.reference_runtime),
                        "candidate_runtime": best.runtime,
                        "reference_runtime": best.reference_runtime,
                        "speedup": best.speedup,
                        "primary_reference": best.primary_reference or result.primary_reference or "torch_eager",
                        "torch_eager_reference_runtime": runtime_for_mode(best.reference_runtimes, "torch_eager", best.reference_runtime),
                        "torch_compile_default_reference_runtime": runtime_for_mode(best.reference_runtimes, "torch_compile_default"),
                        "torch_compile_max_autotune_reference_runtime": runtime_for_mode(best.reference_runtimes, "torch_compile_max_autotune"),
                        "torch_eager_speedup": speedup_for_mode(best.speedups, "torch_eager", best.speedup),
                        "torch_compile_default_speedup": speedup_for_mode(best.speedups, "torch_compile_default"),
                        "torch_compile_max_autotune_speedup": speedup_for_mode(best.speedups, "torch_compile_max_autotune"),
                        "candidate_total_count": stats["total"],
                        "candidate_compile_count": stats["compile"],
                        "candidate_correct_count": stats["correct"],
                        "compile_rate": stats["compile_rate"],
                        "correct_rate": stats["correct_rate"],
                        "failure_stage": best.latest_failure_stage or "none",
                        "failure_type": best.failure_type,
                        "validation_correctness_matches": (validation.get("checks") or {}).get("correctness_matches"),
                        "validation_speed_direction_matches": (validation.get("checks") or {}).get("speed_direction_matches"),
                    }
                )
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
        summary_payload = {
            "manifest": manifest,
            "workflow": workflow,
            "provider": _resolve_provider_name(args, run_name),
            "run_profile": run_name,
            "search_profile": _resolve_search_name(args, run_name),
            "evaluator_profile": _resolve_evaluator_name(args, run_name),
            "measurement_profile": _resolve_measurement_name(args, run_name),
            "agent_provider_profile": _resolve_route_name(args, run_name),
            "deliberation_profile": _resolve_deliberation_name(args, run_name),
            "kernelbench_root": kernelbench_root,
            "output_dir": str(output_root),
            "rows": rows,
            "aggregates": aggregate_batch_rows(rows),
        }
        (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        write_batch_csv(rows, output_root / "summary.csv")
        print(f"summary_json={output_root / 'summary.json'}")
        print(f"summary_csv={output_root / 'summary.csv'}")
        return 0
    finally:
        _close_provider(provider)
        if deliberation_runner is not None:
            deliberation_runner.close()


def _verify_kernelbench(args: argparse.Namespace) -> int:
    validation_path = verify_kernelbench_run(args.path, kernelbench_root=args.kernelbench_root, output_path=args.output)
    payload = load_validation(validation_path)
    print(f"saved={validation_path}")
    if payload is not None:
        print(
            f"validation correct={payload['validation']['correct']} "
            f"direction={payload['validation']['speed_direction']} "
            f"correctness_match={payload['checks']['correctness_matches']} "
            f"direction_match={payload['checks']['speed_direction_matches']}"
        )
    return 0


def _resolve_summary_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find summary.json at: {candidate}")
    return candidate


def _report_paper(args: argparse.Namespace) -> int:
    named_summaries: dict[str, dict[str, Any]] = {}
    for raw_path in args.paths:
        summary_path = _resolve_summary_path(raw_path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        label = summary_path.parent.name if summary_path.parent.name else summary_path.stem
        if label in named_summaries:
            label = f"{label}_{len(named_summaries) + 1}"
        named_summaries[label] = payload
    written = write_paper_summary_report(build_paper_summary_report(named_summaries), args.output_dir, title=args.title)
    print(f"paper_report_json={written['json']}")
    print(f"paper_report_markdown={written['markdown']}")
    return 0


def _save_and_print(result, output_dir: str) -> int:
    run_path = save_run(result, output_dir)
    best = result.nodes[result.best_node_id]
    print(f"saved={run_path}")
    print(f"workflow={result.workflow} task={result.task_name} best_node={result.best_node_id} status={best.status}")
    print(f"run_profile={result.run_profile} search={result.search_profile} evaluator={result.evaluator_profile} measurement={result.measurement_profile}")
    if result.strategy_portfolio is not None:
        print(f"deliberation={result.config.deliberation_profile} strategies={len(result.strategy_portfolio.strategies)} enabled={result.strategy_portfolio.enabled}")
    print(f"best_runtime={shorten_runtime(best.runtime)} reference_runtime={shorten_runtime(best.reference_runtime)} speedup={format_speedup(best.speedup)}")
    return 0


def _show_run(args: argparse.Namespace) -> int:
    result = load_run(args.path)
    best = result.nodes[result.best_node_id]
    print(f"task={result.task_name} workflow={result.workflow} best_node={result.best_node_id} status={best.status}")
    print(f"run_profile={result.run_profile} search={result.search_profile} evaluator={result.evaluator_profile} measurement={result.measurement_profile}")
    print(f"attempts={result.stats.get('attempt_count', 0)} pruned={len(result.pruned_nodes)}")
    print(f"best_runtime={shorten_runtime(best.runtime)} reference_runtime={shorten_runtime(best.reference_runtime)} speedup={format_speedup(best.speedup)}")
    validation = load_validation(args.path)
    if validation:
        print(
            f"validation_correct={validation['validation']['correct']} "
            f"direction_match={validation['checks']['speed_direction_matches']}"
        )
    return 0


def _summarize_runs(args: argparse.Namespace) -> int:
    for raw_path in args.paths:
        result = load_run(raw_path)
        best = result.nodes[result.best_node_id]
        print(
            f"{raw_path}: task={result.task_name} workflow={result.workflow} "
            f"run_profile={result.run_profile} best={result.best_node_id} "
            f"status={best.status} speedup={format_speedup(best.speedup)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-demo":
        return _run_demo(args)
    if args.command == "run-triton":
        return _run_triton(args)
    if args.command == "run-kernelbench":
        return _run_kernelbench(args)
    if args.command == "run-kernelbench-batch":
        return _run_kernelbench_batch(args)
    if args.command == "verify-kernelbench":
        return _verify_kernelbench(args)
    if args.command == "report-paper":
        return _report_paper(args)
    if args.command == "show-run":
        return _show_run(args)
    if args.command == "summarize-runs":
        return _summarize_runs(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
