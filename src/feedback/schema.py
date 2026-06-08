from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AttemptRecord:
    attempt_id: str
    parent_id: str | None
    strategy_name: str | None
    mode: str = "explore"
    mutation_family: str | None = None
    speedup: float | None = None
    parent_speedup: float | None = None
    compile_ok: bool = False
    correct: bool = False
    failure_stage: str | None = None
    failure_type: str | None = None
    single_change_focus: str | None = None


@dataclass
class ChampionState:
    node_id: str | None = None
    speedup: float | None = None
    strategy_name: str | None = None
    mutation_family: str | None = None
    lineage: list[str] = field(default_factory=list)
    recent_positive_mutations: list[dict[str, Any]] = field(default_factory=list)
    recent_negative_mutations: list[dict[str, Any]] = field(default_factory=list)
    plateau_detected: bool = False
    lineage_plateau_depth: int = 0


@dataclass
class FeedbackState:
    total_attempts: int = 0
    compile_rate: float = 0.0
    correct_rate: float = 0.0
    best_speedup: float | None = None
    best_strategy_name: str | None = None
    phase: str = "exploration"
    current_champion_id: str | None = None
    current_champion_speedup: float | None = None
    plateau_detected: bool = False
    recent_improvement_deltas: list[float] = field(default_factory=list)
    recent_regression_deltas: list[float] = field(default_factory=list)
    recent_successful_mutation_families: list[str] = field(default_factory=list)
    recent_failed_mutation_families: list[str] = field(default_factory=list)
    last_mutation_outcome: str | None = None
    recent_attempts: list[AttemptRecord] = field(default_factory=list)
    champion: ChampionState = field(default_factory=ChampionState)


def feedback_state_to_dict(state: FeedbackState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return asdict(state)


def feedback_state_from_dict(payload: dict[str, Any] | None) -> FeedbackState | None:
    if not payload:
        return None
    champion_payload = payload.get("champion") or {}
    return FeedbackState(
        total_attempts=int(payload.get("total_attempts", 0)),
        compile_rate=float(payload.get("compile_rate", 0.0)),
        correct_rate=float(payload.get("correct_rate", 0.0)),
        best_speedup=payload.get("best_speedup"),
        best_strategy_name=payload.get("best_strategy_name"),
        phase=str(payload.get("phase", "exploration")),
        current_champion_id=payload.get("current_champion_id"),
        current_champion_speedup=payload.get("current_champion_speedup"),
        plateau_detected=bool(payload.get("plateau_detected", False)),
        recent_improvement_deltas=[float(item) for item in payload.get("recent_improvement_deltas", [])],
        recent_regression_deltas=[float(item) for item in payload.get("recent_regression_deltas", [])],
        recent_successful_mutation_families=[str(item) for item in payload.get("recent_successful_mutation_families", [])],
        recent_failed_mutation_families=[str(item) for item in payload.get("recent_failed_mutation_families", [])],
        last_mutation_outcome=payload.get("last_mutation_outcome"),
        recent_attempts=[
            AttemptRecord(
                attempt_id=item.get("attempt_id", ""),
                parent_id=item.get("parent_id"),
                strategy_name=item.get("strategy_name"),
                mode=str(item.get("mode", "explore")),
                mutation_family=item.get("mutation_family"),
                speedup=item.get("speedup"),
                parent_speedup=item.get("parent_speedup"),
                compile_ok=bool(item.get("compile_ok", False)),
                correct=bool(item.get("correct", False)),
                failure_stage=item.get("failure_stage"),
                failure_type=item.get("failure_type"),
                single_change_focus=item.get("single_change_focus"),
            )
            for item in payload.get("recent_attempts", [])
        ],
        champion=ChampionState(
            node_id=champion_payload.get("node_id"),
            speedup=champion_payload.get("speedup"),
            strategy_name=champion_payload.get("strategy_name"),
            mutation_family=champion_payload.get("mutation_family"),
            lineage=list(champion_payload.get("lineage", [])),
            recent_positive_mutations=list(champion_payload.get("recent_positive_mutations", [])),
            recent_negative_mutations=list(champion_payload.get("recent_negative_mutations", [])),
            plateau_detected=bool(champion_payload.get("plateau_detected", False)),
            lineage_plateau_depth=int(champion_payload.get("lineage_plateau_depth", 0)),
        ),
    )
