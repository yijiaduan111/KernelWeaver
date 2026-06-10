"""Merge and rank model-proposed optimization strategies."""

from __future__ import annotations

import re
from collections import OrderedDict

from .schema import DeliberationStrategy, ModelProposal, ModelReview, StrategyPortfolio


def merge_strategy_proposals(
    proposals: list[ModelProposal],
    max_strategies: int,
    mode: str = "multi_model_v0",
) -> StrategyPortfolio:
    """Deduplicate proposals and keep a compact ranked strategy portfolio."""

    providers = [proposal.provider_name for proposal in proposals]
    proposal_errors = {proposal.provider_name: proposal.error or "unknown" for proposal in proposals if proposal.status != "ok"}
    merged: OrderedDict[str, DeliberationStrategy] = OrderedDict()
    for proposal in proposals:
        for strategy in proposal.strategies:
            key = _strategy_key(strategy)
            if key not in merged:
                merged[key] = DeliberationStrategy(
                    strategy_id="",
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
                continue
            current = merged[key]
            current.target_anchors = _dedupe([*current.target_anchors, *strategy.target_anchors])
            current.implementation_hints = _dedupe([*current.implementation_hints, *strategy.implementation_hints])[:8]
            current.risk_notes = _dedupe([*current.risk_notes, *strategy.risk_notes])[:8]
            current.memory_methods = _dedupe([*current.memory_methods, *strategy.memory_methods])[:4]
            current.mutation_axes = _dedupe([*current.mutation_axes, *strategy.mutation_axes])[:6]
            current.forbidden_patterns = _dedupe([*current.forbidden_patterns, *strategy.forbidden_patterns])[:6]
            current.source_models = _dedupe([*current.source_models, *strategy.source_models])
            current.model_scores.update(strategy.model_scores)
            current.review_notes.update(strategy.review_notes)
            current.priority = max(current.priority, strategy.priority)
    ranked = sorted(merged.values(), key=_rank_key, reverse=True)[: max(0, max_strategies)]
    for index, strategy in enumerate(ranked, start=1):
        strategy.strategy_id = f"strategy_{index:02d}"
    return StrategyPortfolio(
        enabled=bool(ranked),
        mode=mode,
        max_strategies=max_strategies,
        providers=providers,
        strategies=ranked,
        proposal_errors=proposal_errors,
    )


def apply_strategy_reviews(portfolio: StrategyPortfolio, reviews: list[ModelReview]) -> StrategyPortfolio:
    by_id = {strategy.strategy_id: strategy for strategy in portfolio.strategies}
    for review in reviews:
        if review.status != "ok":
            portfolio.review_errors[review.provider_name] = review.error or "unknown"
            continue
        for strategy_id, score in review.scores.items():
            if strategy_id in by_id:
                by_id[strategy_id].model_scores[review.provider_name] = float(score)
        for strategy_id, note in review.notes.items():
            if strategy_id in by_id and note:
                by_id[strategy_id].review_notes[review.provider_name] = str(note)
    portfolio.strategies.sort(key=_rank_key, reverse=True)
    return portfolio


def _strategy_key(strategy: DeliberationStrategy) -> str:
    anchors = ",".join(sorted(strategy.target_anchors[:3]))
    text = _normalize(f"{strategy.intent} {strategy.summary}")
    words = " ".join(text.split()[:10])
    return f"{anchors}|{words}"


def _rank_key(strategy: DeliberationStrategy) -> tuple[float, float, int, int]:
    support = len(strategy.source_models)
    avg_score = sum(strategy.model_scores.values()) / len(strategy.model_scores) if strategy.model_scores else 0.0
    anchors = len(strategy.target_anchors)
    return (support, avg_score, strategy.priority, anchors)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", text.lower()).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
