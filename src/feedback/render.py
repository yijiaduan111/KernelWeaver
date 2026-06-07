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
        "recent_failure_types": list(state.recent_failure_types),
        "strategy_summaries": [
            {
                "strategy_name": item.strategy_name,
                "total_attempts": item.total_attempts,
                "compile_success": item.compile_success,
                "correct_success": item.correct_success,
                "best_speedup": round(item.best_speedup, 3) if item.best_speedup is not None else None,
                "avg_delta_vs_parent": round(item.avg_delta_vs_parent, 3) if item.avg_delta_vs_parent is not None else None,
                "dominant_failure_type": item.dominant_failure_type,
            }
            for item in state.strategy_summaries
        ],
    }
