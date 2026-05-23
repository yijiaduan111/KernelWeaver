"""Prompt rendering helpers for strategy portfolios."""

from __future__ import annotations

from typing import Any

from .schema import StrategyPortfolio


def strategy_portfolio_to_prompt_dict(portfolio: StrategyPortfolio | None, max_strategies: int | None = None) -> dict[str, Any] | None:
    if portfolio is None or not portfolio.enabled or not portfolio.strategies:
        return None
    limit = max_strategies or portfolio.max_strategies
    return {
        "mode": portfolio.mode,
        "providers": list(portfolio.providers),
        "strategy_ids": [strategy.strategy_id for strategy in portfolio.strategies[:limit]],
        "strategies": [
            {
                "strategy_id": strategy.strategy_id,
                "intent": strategy.intent,
                "summary": strategy.summary,
                "target_anchors": strategy.target_anchors[:6],
                "implementation_hints": strategy.implementation_hints[:6],
                "expected_gain": strategy.expected_gain,
                "risk_notes": strategy.risk_notes[:5],
                "source_models": strategy.source_models,
                "model_scores": strategy.model_scores,
            }
            for strategy in portfolio.strategies[:limit]
        ],
        "proposal_errors": portfolio.proposal_errors,
        "review_errors": portfolio.review_errors,
    }
