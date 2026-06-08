from __future__ import annotations

from .schema import FeedbackState


def feedback_state_to_prompt_dict(state: FeedbackState | None) -> dict | None:
    if state is None:
        return None
    return {
        "phase": state.phase,
        "total_attempts": state.total_attempts,
        "compile_rate": round(state.compile_rate, 3),
        "correct_rate": round(state.correct_rate, 3),
        "best_speedup": round(state.best_speedup, 3) if state.best_speedup is not None else None,
        "best_strategy_name": state.best_strategy_name,
        "current_champion_id": state.current_champion_id,
        "current_champion_speedup": round(state.current_champion_speedup, 3) if state.current_champion_speedup is not None else None,
        "plateau_detected": state.plateau_detected,
        "recent_improvement_deltas": [round(item, 3) for item in state.recent_improvement_deltas],
        "recent_regression_deltas": [round(item, 3) for item in state.recent_regression_deltas],
        "recent_successful_mutation_families": list(state.recent_successful_mutation_families),
        "recent_failed_mutation_families": list(state.recent_failed_mutation_families),
        "last_mutation_outcome": state.last_mutation_outcome,
        "champion": {
            "node_id": state.champion.node_id,
            "speedup": round(state.champion.speedup, 3) if state.champion.speedup is not None else None,
            "strategy_name": state.champion.strategy_name,
            "mutation_family": state.champion.mutation_family,
            "lineage": list(state.champion.lineage),
            "recent_positive_mutations": list(state.champion.recent_positive_mutations),
            "recent_negative_mutations": list(state.champion.recent_negative_mutations),
            "plateau_detected": state.champion.plateau_detected,
            "lineage_plateau_depth": state.champion.lineage_plateau_depth,
        },
    }
