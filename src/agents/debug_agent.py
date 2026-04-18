"""Debug agent wrapper."""

from __future__ import annotations

from .base import BaseAgent
from ..models import AgentContext, SearchNode, TaskSpec


class DebugAgent(BaseAgent):
    def run(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> str:
        return self.provider.debug_code(task, node, context)
