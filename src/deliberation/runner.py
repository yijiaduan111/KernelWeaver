"""Task-level multi-model deliberation runner."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..models import StarkConfig, TaskSpec
from ..semantics import semantic_profile_to_prompt_dict
from ..utils import extract_anchor_names
from .merge import apply_strategy_reviews, merge_strategy_proposals
from .schema import DeliberationStrategy, ModelProposal, ModelReview, StrategyPortfolio
from .render import strategy_portfolio_to_prompt_dict
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
                    if phase == "propose":
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

    def _review(self, provider_name: str, provider: Any, task: TaskSpec, portfolio: StrategyPortfolio) -> ModelReview:
        try:
            text = provider.generate_text(
                system_prompt=_review_system_prompt(),
                user_payload={
                    "provider_name": provider_name,
                    "task_name": task.name,
                    "backend": task.backend,
                    "semantic_profile": semantic_profile_to_prompt_dict(task.semantic_profile),
                    "strategy_portfolio": strategy_portfolio_to_prompt_dict(portfolio),
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
        "Use only provided anchors. Return JSON only. "
        f"Return at most {strategies_per_model} strategies with schema: "
        '{"strategies":[{"intent":"...","summary":"...","target_anchors":["..."],'
        '"implementation_hints":["..."],"expected_gain":"...","risk_notes":["..."],"score":1-5}]}.'
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


def _parse_strategies(data: dict[str, Any], provider_name: str, task: TaskSpec, limit: int) -> list[DeliberationStrategy]:
    available = set(extract_anchor_names(task.source_code))
    strategies: list[DeliberationStrategy] = []
    for item in data.get("strategies", []) or []:
        if not isinstance(item, dict):
            continue
        target_anchors = [str(anchor).strip() for anchor in item.get("target_anchors", []) if str(anchor).strip() in available]
        if not target_anchors and task.semantic_profile is not None:
            target_anchors = [anchor for anchor in task.semantic_profile.recommended_anchors if anchor in available][:2]
        strategies.append(
            DeliberationStrategy(
                strategy_id="",
                intent=str(item.get("intent") or item.get("name") or "optimization_strategy"),
                summary=str(item.get("summary") or item.get("strategy_summary") or "Model-proposed optimization strategy."),
                target_anchors=target_anchors,
                implementation_hints=_string_list(item.get("implementation_hints") or item.get("hints")),
                expected_gain=str(item.get("expected_gain") or "unknown"),
                risk_notes=_string_list(item.get("risk_notes") or item.get("risks")),
                source_models=[provider_name],
                model_scores={provider_name: _score(item.get("score", 3))},
                priority=int(_score(item.get("score", 3))),
            )
        )
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


def _result_detail(result: Any) -> str:
    if isinstance(result, ModelProposal):
        return f"status={result.status} strategies={len(result.strategies)}"
    if isinstance(result, ModelReview):
        return f"status={result.status} scores={len(result.scores)}"
    return type(result).__name__
