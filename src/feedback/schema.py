from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AttemptRecord:
    attempt_id: str
    parent_id: str | None
    strategy_name: str | None
    anchor_names: list[str] = field(default_factory=list)
    origin: str = "plan_code"
    compile_ok: bool = False
    correct: bool = False
    speedup: float | None = None
    failure_stage: str | None = None
    failure_type: str | None = None
    parent_speedup: float | None = None
    node_status: str | None = None


@dataclass
class StrategyAttemptSummary:
    strategy_name: str
    total_attempts: int
    compile_success: int
    correct_success: int
    best_speedup: float | None
    avg_delta_vs_parent: float | None
    dominant_failure_type: str | None


@dataclass
class FeedbackState:
    strategy_summaries: list[StrategyAttemptSummary] = field(default_factory=list)
    total_attempts: int = 0
    compile_rate: float = 0.0
    correct_rate: float = 0.0
    best_speedup: float | None = None
    best_strategy_name: str | None = None
    recent_failure_types: list[str] = field(default_factory=list)
    recent_attempts: list[AttemptRecord] = field(default_factory=list)
    phase: str = "exploration"


def feedback_state_to_dict(state: FeedbackState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return asdict(state)


def feedback_state_from_dict(payload: dict[str, Any] | None) -> FeedbackState | None:
    if not payload:
        return None
    return FeedbackState(
        strategy_summaries=[
            StrategyAttemptSummary(
                strategy_name=item.get("strategy_name", ""),
                total_attempts=int(item.get("total_attempts", 0)),
                compile_success=int(item.get("compile_success", 0)),
                correct_success=int(item.get("correct_success", 0)),
                best_speedup=item.get("best_speedup"),
                avg_delta_vs_parent=item.get("avg_delta_vs_parent"),
                dominant_failure_type=item.get("dominant_failure_type"),
            )
            for item in payload.get("strategy_summaries", [])
        ],
        total_attempts=int(payload.get("total_attempts", 0)),
        compile_rate=float(payload.get("compile_rate", 0.0)),
        correct_rate=float(payload.get("correct_rate", 0.0)),
        best_speedup=payload.get("best_speedup"),
        best_strategy_name=payload.get("best_strategy_name"),
        recent_failure_types=list(payload.get("recent_failure_types", [])),
        recent_attempts=[
            AttemptRecord(
                attempt_id=item.get("attempt_id", ""),
                parent_id=item.get("parent_id"),
                strategy_name=item.get("strategy_name"),
                anchor_names=list(item.get("anchor_names", [])),
                origin=item.get("origin", "plan_code"),
                compile_ok=bool(item.get("compile_ok", False)),
                correct=bool(item.get("correct", False)),
                speedup=item.get("speedup"),
                failure_stage=item.get("failure_stage"),
                failure_type=item.get("failure_type"),
                parent_speedup=item.get("parent_speedup"),
                node_status=item.get("node_status"),
            )
            for item in payload.get("recent_attempts", [])
        ],
        phase=str(payload.get("phase", "exploration")),
    )
