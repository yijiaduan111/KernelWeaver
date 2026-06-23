"""Task-level multi-model deliberation runner."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..diagnostics import ncu_profile_to_prompt_dict
from ..models import StarkConfig, TaskSpec
from ..semantics import semantic_profile_to_prompt_dict
from ..utils import extract_anchor_names
from .merge import apply_strategy_reviews, merge_strategy_proposals
from .render import strategy_portfolio_to_prompt_dict
from .schema import DeliberationStrategy, ModelProposal, ModelReview, StrategyPortfolio
from .telemetry import DeliberationEvent


class MultiModelDeliberationRunner:
    """Collect strategy priors from several model providers before search."""

    def __init__(
        self,
        providers: dict[str, Any],
        max_strategies: int = 10,
        strategies_per_model: int = 4,
        proposal_temperature: float = 0.4,
        review_temperature: float = 0.1,
        mode: str = "multi_model_v0",
    ) -> None:
        self.providers = providers
        self.max_strategies = max_strategies
        self.strategies_per_model = strategies_per_model
        self.proposal_temperature = proposal_temperature
        self.review_temperature = review_temperature
        self.mode = mode
        self.last_events: list[DeliberationEvent] = []

    def run(self, task: TaskSpec, config: StarkConfig) -> StrategyPortfolio:
        del config
        self.last_events = []
        proposals = self._collect_parallel(
            phase="propose",
            action=lambda provider_name, provider: self._propose(provider_name, provider, task),
        )
        portfolio = merge_strategy_proposals(proposals, max_strategies=self.max_strategies, mode=self.mode)
        if not portfolio.strategies:
            portfolio.enabled = False
            return portfolio
        reviews = self._collect_parallel(
            phase="review",
            action=lambda provider_name, provider: self._review(provider_name, provider, task, portfolio),
        )
        return apply_strategy_reviews(portfolio, reviews)

    def run_upgrade(
        self,
        task: TaskSpec,
        config: StarkConfig,
        feedback_state,
        existing_portfolio: StrategyPortfolio,
        current_round: int,
        *,
        champion_summary: dict[str, Any] | None = None,
        champion_code: str | None = None,
    ) -> StrategyPortfolio:
        if existing_portfolio is None or not existing_portfolio.strategies:
            return self.run(task, config)
        del config
        self.last_events = []
        proposals = self._collect_parallel(
            phase="propose_upgrade",
            action=lambda provider_name, provider: self._propose_upgrade(
                provider_name,
                provider,
                task,
                feedback_state,
                existing_portfolio,
                current_round,
                champion_summary=champion_summary,
                champion_code=champion_code,
            ),
        )
        upgrade_pool = merge_strategy_proposals(proposals, max_strategies=self.max_strategies, mode=self.mode)
        if not upgrade_pool.strategies:
            return existing_portfolio
        upgrade_pool.strategies = _filter_novel_strategies(existing_portfolio, upgrade_pool.strategies)
        if not upgrade_pool.strategies:
            return existing_portfolio
        upgrade_pool.enabled = True
        upgrade_pool.deliberation_round = current_round
        reviews = self._collect_parallel(
            phase="review_upgrade",
            action=lambda provider_name, provider: self._review(provider_name, provider, task, upgrade_pool),
        )
        reviewed = apply_strategy_reviews(upgrade_pool, reviews)
        if not reviewed.strategies:
            return existing_portfolio
        return _merge_strategy_portfolios(existing_portfolio, reviewed, current_round)

    def _collect_parallel(self, phase: str, action):
        provider_items = list(self.providers.items())
        if not provider_items:
            return []
        results_by_name: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(provider_items)), thread_name_prefix=f"delib_{phase}") as executor:
            futures = {}
            for provider_name, provider in provider_items:
                self.last_events.append(DeliberationEvent(phase=phase, provider_name=provider_name, status="start"))
                futures[executor.submit(self._timed_action, provider_name, provider, action)] = provider_name
            for future in as_completed(futures):
                provider_name = futures[future]
                try:
                    result, elapsed_seconds = future.result()
                    self.last_events.append(
                        DeliberationEvent(
                            phase=phase,
                            provider_name=provider_name,
                            status="ok",
                            elapsed_seconds=elapsed_seconds,
                            detail=_result_detail(result),
                        )
                    )
                    results_by_name[provider_name] = result
                except Exception as exc:
                    self.last_events.append(
                        DeliberationEvent(
                            phase=phase,
                            provider_name=provider_name,
                            status="error",
                            detail=str(exc),
                        )
                    )
                    if phase.startswith("propose"):
                        results_by_name[provider_name] = ModelProposal(provider_name=provider_name, status="error", error=str(exc))
                    else:
                        results_by_name[provider_name] = ModelReview(provider_name=provider_name, status="error", error=str(exc))
        return [results_by_name[provider_name] for provider_name, _provider in provider_items]

    @staticmethod
    def _timed_action(provider_name: str, provider: Any, action):
        started = time.time()
        result = action(provider_name, provider)
        return result, time.time() - started

    def close(self) -> None:
        seen: set[int] = set()
        for provider in self.providers.values():
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                close_fn()

    def _propose(self, provider_name: str, provider: Any, task: TaskSpec) -> ModelProposal:
        try:
            text = provider.generate_text(
                system_prompt=_proposal_system_prompt(self.strategies_per_model),
                user_payload=_proposal_payload(task, provider_name, self.strategies_per_model),
                temperature=self.proposal_temperature,
                purpose="deliberation_propose",
            )
            data = _parse_json_object(text)
            strategies = _parse_strategies(data, provider_name, task, self.strategies_per_model)
            return ModelProposal(provider_name=provider_name, strategies=strategies)
        except Exception as exc:
            return ModelProposal(provider_name=provider_name, status="error", error=str(exc))

    def _propose_upgrade(
        self,
        provider_name: str,
        provider: Any,
        task: TaskSpec,
        feedback_state,
        existing_portfolio: StrategyPortfolio,
        current_round: int,
        *,
        champion_summary: dict[str, Any] | None = None,
        champion_code: str | None = None,
    ) -> ModelProposal:
        try:
            text = provider.generate_text(
                system_prompt=_upgrade_proposal_system_prompt(self.strategies_per_model, current_round),
                user_payload=_upgrade_proposal_payload(
                    task,
                    provider_name,
                    self.strategies_per_model,
                    feedback_state,
                    existing_portfolio,
                    current_round,
                    champion_summary=champion_summary,
                    champion_code=champion_code,
                ),
                temperature=self.proposal_temperature,
                purpose="deliberation_propose_upgrade",
            )
            data = _parse_json_object(text)
            strategies = _parse_strategies(data, provider_name, task, self.strategies_per_model)
            return ModelProposal(provider_name=provider_name, strategies=strategies)
        except Exception as exc:
            return ModelProposal(provider_name=provider_name, status="error", error=str(exc))

    def _review(self, provider_name: str, provider: Any, task: TaskSpec, portfolio: StrategyPortfolio) -> ModelReview:
        try:
            text = provider.generate_text(
                system_prompt=_review_system_prompt(),
                user_payload={
                    "provider_name": provider_name,
                    "task_name": task.name,
                    "backend": task.backend,
                    "semantic_profile": semantic_profile_to_prompt_dict(task.semantic_profile),
                    "strategy_portfolio": strategy_portfolio_to_prompt_dict(
                        portfolio,
                        max_strategies=max(len(portfolio.strategies), portfolio.max_strategies),
                    ),
                },
                temperature=self.review_temperature,
                purpose="deliberation_review",
            )
            data = _parse_json_object(text)
            scores: dict[str, float] = {}
            notes: dict[str, str] = {}
            for item in data.get("scores", []) or []:
                if not isinstance(item, dict):
                    continue
                strategy_id = str(item.get("strategy_id", "")).strip()
                if not strategy_id:
                    continue
                scores[strategy_id] = _score(item.get("score", 0))
                note = str(item.get("notes", "")).strip()
                if note:
                    notes[strategy_id] = note
            return ModelReview(provider_name=provider_name, scores=scores, notes=notes)
        except Exception as exc:
            return ModelReview(provider_name=provider_name, status="error", error=str(exc))


def _proposal_system_prompt(strategies_per_model: int) -> str:
    return (
        "You are one independent model in a multi-model kernel optimization deliberation. "
        "Propose backend-specific optimization strategies, but do not write code. "
        "If root_ncu_profile is provided, use its raw_metrics as runtime evidence about the current root candidate when deciding strategies. "
        "Use only provided anchors. Return JSON only. "
        f"Return at most {strategies_per_model} strategies with schema: "
        '{"strategies":[{"intent":"...","summary":"...","target_anchors":["..."],'
        '"implementation_hints":["..."],"expected_gain":"...","risk_notes":["..."],'
        '"memory_methods":["..."],"mutation_axes":["..."],"forbidden_patterns":["..."],"score":1-5}]}.'
    )


def _upgrade_proposal_system_prompt(strategies_per_model: int, current_round: int) -> str:
    return (
        f"You are one model in a multi-model kernel optimization deliberation, round {current_round}. "
        "Previous strategies have already been explored and the search is plateauing. "
        "You will receive the current champion summary, a compact champion code excerpt, the existing strategy portfolio, and recent search outcomes. "
        "Propose strategies that are materially different from the explored directions. "
        "Either refine the current champion with one new hypothesis, or propose a genuinely different algorithmic direction. "
        "Do not repeat strategies that already exist in the portfolio or strategies that have clearly plateaued without a new concrete fix. "
        "If a strategy family has compile-failed repeatedly, either repair the root cause concretely or avoid that family. "
        "If root_ncu_profile is provided, use its raw_metrics as runtime evidence about the current root candidate when proposing new strategies. "
        "Use only provided anchors. Return JSON only. "
        f"Return at most {strategies_per_model} strategies with schema: "
        '{"strategies":[{"intent":"...","summary":"...","target_anchors":["..."],'
        '"implementation_hints":["..."],"expected_gain":"...","risk_notes":["..."],'
        '"memory_methods":["..."],"mutation_axes":["..."],"forbidden_patterns":["..."],"score":1-5}]}.'
    )


def _review_system_prompt() -> str:
    return (
        "You are reviewing a merged strategy portfolio for a GPU kernel optimization search. "
        "Score each strategy from 1 to 5 for likely usefulness and risk-adjusted value. "
        "Return JSON only with schema: "
        '{"scores":[{"strategy_id":"strategy_01","score":4,"notes":"..."}]}.'
    )


def _proposal_payload(task: TaskSpec, provider_name: str, strategies_per_model: int) -> dict[str, Any]:
    anchors = extract_anchor_names(task.source_code)
    return {
        "provider_name": provider_name,
        "max_strategies": strategies_per_model,
        "task_name": task.name,
        "task_description": task.description,
        "backend": task.backend,
        "benchmark_family": task.benchmark_family,
        "level": task.level,
        "problem_id": task.problem_id,
        "available_anchors": anchors,
        "semantic_profile": semantic_profile_to_prompt_dict(task.semantic_profile),
        "root_ncu_profile": ncu_profile_to_prompt_dict(
            getattr(getattr(task, "diagnostics_profile", None), "ncu_profile", None)
        ),
        "execution_facts": task.execution_facts.to_prompt_dict() if task.execution_facts else None,
        "grounded_regions": [
            {
                "anchor_name": region.anchor_name,
                "region_role": region.region_role,
                "source_excerpt": region.source_excerpt,
            }
            for region in task.grounded_regions
        ],
        "current_scaffold": task.source_code,
    }


def _upgrade_proposal_payload(
    task: TaskSpec,
    provider_name: str,
    strategies_per_model: int,
    feedback_state,
    existing_portfolio: StrategyPortfolio,
    current_round: int,
    *,
    champion_summary: dict[str, Any] | None = None,
    champion_code: str | None = None,
) -> dict[str, Any]:
    base = _proposal_payload(task, provider_name, strategies_per_model)
    recent_attempts = list(getattr(feedback_state, "recent_attempts", []) or [])
    strategy_outcomes: list[dict[str, Any]] = []
    for strategy in existing_portfolio.strategies:
        related = [attempt for attempt in recent_attempts if getattr(attempt, "strategy_name", None) == strategy.strategy_id]
        speedups = [float(attempt.speedup) for attempt in related if isinstance(getattr(attempt, "speedup", None), (int, float))]
        strategy_outcomes.append(
            {
                "strategy_id": strategy.strategy_id,
                "intent": strategy.intent,
                "summary": strategy.summary,
                "best_speedup": max(speedups) if speedups else None,
                "attempt_count": len(related),
                "compile_fail_count": sum(1 for attempt in related if not bool(getattr(attempt, "compile_ok", False))),
                "correctness_fail_count": sum(
                    1
                    for attempt in related
                    if bool(getattr(attempt, "compile_ok", False)) and not bool(getattr(attempt, "correct", False))
                ),
                "mutation_axes": list(strategy.mutation_axes[:4]),
                "risk_notes": list(strategy.risk_notes[:4]),
            }
        )
    base["existing_strategy_portfolio"] = strategy_portfolio_to_prompt_dict(
        existing_portfolio,
        max_strategies=max(len(existing_portfolio.strategies), existing_portfolio.max_strategies),
    )
    base["iteration_context"] = {
        "current_round": current_round,
        "previous_round": getattr(existing_portfolio, "deliberation_round", 1),
        "plateau_detected": bool(getattr(feedback_state, "plateau_detected", False)),
        "champion_speedup": getattr(feedback_state, "current_champion_speedup", None),
        "champion_strategy": getattr(feedback_state, "best_strategy_name", None),
        "champion_summary": champion_summary,
        "champion_code_excerpt": _trim_code_excerpt(champion_code),
        "strategy_outcomes": strategy_outcomes,
        "recent_attempts": _recent_attempts_payload(recent_attempts),
        "recent_improvement_deltas": list(getattr(feedback_state, "recent_improvement_deltas", []) or []),
        "recent_regression_deltas": list(getattr(feedback_state, "recent_regression_deltas", []) or []),
        "recent_successful_mutation_families": list(getattr(feedback_state, "recent_successful_mutation_families", []) or []),
        "recent_failed_mutation_families": list(getattr(feedback_state, "recent_failed_mutation_families", []) or []),
        "recent_positive_mutations": list(getattr(getattr(feedback_state, "champion", None), "recent_positive_mutations", []) or []),
        "recent_negative_mutations": list(getattr(getattr(feedback_state, "champion", None), "recent_negative_mutations", []) or []),
        "instruction": (
            "The search has plateaued. Propose new strategies that either push past the current ceiling or introduce a different optimization family. "
            "Do not repeat the existing portfolio without a new concrete mechanism."
        ),
    }
    return base


def _recent_attempts_payload(recent_attempts: list[Any], limit: int = 8) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for attempt in recent_attempts[-limit:]:
        payload.append(
            {
                "attempt_id": getattr(attempt, "attempt_id", None),
                "parent_id": getattr(attempt, "parent_id", None),
                "strategy_name": getattr(attempt, "strategy_name", None),
                "mode": getattr(attempt, "mode", None),
                "mutation_family": getattr(attempt, "mutation_family", None),
                "single_change_focus": getattr(attempt, "single_change_focus", None),
                "speedup": getattr(attempt, "speedup", None),
                "parent_speedup": getattr(attempt, "parent_speedup", None),
                "compile_ok": bool(getattr(attempt, "compile_ok", False)),
                "correct": bool(getattr(attempt, "correct", False)),
                "failure_stage": getattr(attempt, "failure_stage", None),
                "failure_type": getattr(attempt, "failure_type", None),
            }
        )
    return payload


def _parse_strategies(data: dict[str, Any], provider_name: str, task: TaskSpec, limit: int) -> list[DeliberationStrategy]:
    available = set(extract_anchor_names(task.source_code))
    strategies: list[DeliberationStrategy] = []
    for item in data.get("strategies", []) or []:
        if not isinstance(item, dict):
            continue
        target_anchors = [str(anchor).strip() for anchor in item.get("target_anchors", []) if str(anchor).strip() in available]
        if not target_anchors and task.semantic_profile is not None:
            target_anchors = [anchor for anchor in task.semantic_profile.recommended_anchors if anchor in available][:2]
        memory_methods = _string_list(item.get("memory_methods") or item.get("method_ids"))[:2]
        strategy = DeliberationStrategy(
            strategy_id="",
            intent=str(item.get("intent") or item.get("name") or "optimization_strategy"),
            summary=str(item.get("summary") or item.get("strategy_summary") or "Model-proposed optimization strategy."),
            target_anchors=target_anchors,
            implementation_hints=_string_list(item.get("implementation_hints") or item.get("hints")),
            expected_gain=str(item.get("expected_gain") or "unknown"),
            risk_notes=_string_list(item.get("risk_notes") or item.get("risks")),
            memory_methods=memory_methods[:2],
            mutation_axes=_string_list(item.get("mutation_axes") or item.get("mutation_focuses") or item.get("focuses")),
            forbidden_patterns=_string_list(item.get("forbidden_patterns") or item.get("anti_patterns")),
            source_models=[provider_name],
            model_scores={provider_name: _score(item.get("score", 3))},
            priority=int(_score(item.get("score", 3))),
        )
        strategies.append(strategy)
        if len(strategies) >= limit:
            break
    return strategies


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Deliberation response must be a JSON object.")
    return payload


def _strip_code_fences(text: str) -> str:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _score(value: Any) -> float:
    try:
        return max(0.0, min(5.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _trim_code_excerpt(code: str | None, max_chars: int = 2400) -> str | None:
    if not code:
        return None
    text = str(code).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", str(text).lower()).strip()


def _strategy_identity(strategy: DeliberationStrategy) -> str:
    anchors = ",".join(sorted(strategy.target_anchors[:3]))
    words = " ".join(_normalize_text(f"{strategy.intent} {strategy.summary}").split()[:10])
    return f"{anchors}|{words}"


def _filter_novel_strategies(existing_portfolio: StrategyPortfolio, strategies: list[DeliberationStrategy]) -> list[DeliberationStrategy]:
    existing = {_strategy_identity(strategy) for strategy in existing_portfolio.strategies}
    filtered: list[DeliberationStrategy] = []
    seen_new: set[str] = set()
    for strategy in strategies:
        identity = _strategy_identity(strategy)
        if identity in existing or identity in seen_new:
            continue
        seen_new.add(identity)
        filtered.append(strategy)
    return filtered


def _clone_strategy(strategy: DeliberationStrategy) -> DeliberationStrategy:
    return DeliberationStrategy(
        strategy_id=strategy.strategy_id,
        intent=strategy.intent,
        summary=strategy.summary,
        target_anchors=list(strategy.target_anchors),
        implementation_hints=list(strategy.implementation_hints),
        expected_gain=strategy.expected_gain,
        risk_notes=list(strategy.risk_notes),
        memory_methods=list(strategy.memory_methods),
        mutation_axes=list(strategy.mutation_axes),
        forbidden_patterns=list(strategy.forbidden_patterns),
        source_models=list(strategy.source_models),
        model_scores=dict(strategy.model_scores),
        review_notes=dict(strategy.review_notes),
        priority=strategy.priority,
    )


def _merge_strategy_portfolios(
    existing_portfolio: StrategyPortfolio,
    upgraded_portfolio: StrategyPortfolio,
    current_round: int,
) -> StrategyPortfolio:
    strategies = [_clone_strategy(strategy) for strategy in existing_portfolio.strategies]
    strategies.extend(_clone_strategy(strategy) for strategy in upgraded_portfolio.strategies)
    for index, strategy in enumerate(strategies, start=1):
        strategy.strategy_id = f"strategy_{index:02d}"
    providers = list(dict.fromkeys([*existing_portfolio.providers, *upgraded_portfolio.providers]))
    proposal_errors = dict(existing_portfolio.proposal_errors)
    proposal_errors.update(upgraded_portfolio.proposal_errors)
    review_errors = dict(existing_portfolio.review_errors)
    review_errors.update(upgraded_portfolio.review_errors)
    return StrategyPortfolio(
        enabled=bool(strategies),
        mode=existing_portfolio.mode,
        max_strategies=max(len(strategies), existing_portfolio.max_strategies, upgraded_portfolio.max_strategies),
        providers=providers,
        strategies=strategies,
        proposal_errors=proposal_errors,
        review_errors=review_errors,
        deliberation_round=current_round,
    )


def _result_detail(result: Any) -> str:
    if isinstance(result, ModelProposal):
        return f"status={result.status} strategies={len(result.strategies)}"
    if isinstance(result, ModelReview):
        return f"status={result.status} scores={len(result.scores)}"
    return type(result).__name__
