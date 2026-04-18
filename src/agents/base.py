"""Shared base class for role-specific agent wrappers."""

from __future__ import annotations

from ..providers import AgentProvider


class BaseAgent:
    """Small wrapper that delegates work to one provider instance."""

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider
