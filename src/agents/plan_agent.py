"""Plan agent wrapper."""

from __future__ import annotations

from .base import BaseAgent
from ..models import AgentContext, PlanProposal, SearchNode, TaskSpec


class PlanAgent(BaseAgent):
    def run(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> PlanProposal:
        return self.provider.propose_plan(task, node, context)
