"""Data structures for task-level multi-model strategy deliberation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DeliberationStrategy:
    """One backend-neutral optimization strategy proposed before search."""

    strategy_id: str
    intent: str
    summary: str
    target_anchors: list[str] = field(default_factory=list)
    implementation_hints: list[str] = field(default_factory=list)
    expected_gain: str = "unknown"
    risk_notes: list[str] = field(default_factory=list)
    source_models: list[str] = field(default_factory=list)
    model_scores: dict[str, float] = field(default_factory=dict)
    review_notes: dict[str, str] = field(default_factory=dict)
    priority: int = 3


@dataclass
class ModelProposal:
    """Strategies produced by one model during the proposal phase."""

    provider_name: str
    status: str = "ok"
    strategies: list[DeliberationStrategy] = field(default_factory=list)
    error: str | None = None


@dataclass
class ModelReview:
    """Scores assigned by one model to the merged strategy pool."""

    provider_name: str
    status: str = "ok"
    scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class StrategyPortfolio:
    """Task-level strategy priors consumed by the PlanAgent search loop."""

    enabled: bool = True
    mode: str = "multi_model_v0"
    max_strategies: int = 10
    providers: list[str] = field(default_factory=list)
    strategies: list[DeliberationStrategy] = field(default_factory=list)
    proposal_errors: dict[str, str] = field(default_factory=dict)
    review_errors: dict[str, str] = field(default_factory=dict)


def strategy_portfolio_to_dict(portfolio: StrategyPortfolio | None) -> dict[str, Any] | None:
    if portfolio is None:
        return None
    return asdict(portfolio)


def strategy_portfolio_from_dict(payload: dict[str, Any] | None) -> StrategyPortfolio | None:
    if not payload:
        return None
    strategies = [
        DeliberationStrategy(
            strategy_id=str(item.get("strategy_id", "")),
            intent=str(item.get("intent", "unknown")),
            summary=str(item.get("summary", "")),
            target_anchors=list(item.get("target_anchors") or []),
            implementation_hints=list(item.get("implementation_hints") or []),
            expected_gain=str(item.get("expected_gain", "unknown")),
            risk_notes=list(item.get("risk_notes") or []),
            source_models=list(item.get("source_models") or []),
            model_scores={str(key): float(value) for key, value in dict(item.get("model_scores") or {}).items()},
            review_notes={str(key): str(value) for key, value in dict(item.get("review_notes") or {}).items()},
            priority=int(item.get("priority", 3)),
        )
        for item in payload.get("strategies", [])
        if isinstance(item, dict)
    ]
    return StrategyPortfolio(
        enabled=bool(payload.get("enabled", True)),
        mode=str(payload.get("mode", "multi_model_v0")),
        max_strategies=int(payload.get("max_strategies", 10)),
        providers=list(payload.get("providers") or []),
        strategies=strategies,
        proposal_errors={str(key): str(value) for key, value in dict(payload.get("proposal_errors") or {}).items()},
        review_errors={str(key): str(value) for key, value in dict(payload.get("review_errors") or {}).items()},
    )
