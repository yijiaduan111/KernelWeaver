"""Feedback helpers."""

from .render import feedback_state_to_prompt_dict
from .schema import (
    AttemptRecord,
    ChampionState,
    FeedbackState,
    feedback_state_from_dict,
    feedback_state_to_dict,
)

__all__ = [
    "AttemptRecord",
    "ChampionState",
    "FeedbackState",
    "feedback_state_from_dict",
    "feedback_state_to_dict",
    "feedback_state_to_prompt_dict",
    "collect_feedback_state",
]


def collect_feedback_state(*args, **kwargs):
    from .collector import collect_feedback_state as _collect_feedback_state

    return _collect_feedback_state(*args, **kwargs)
