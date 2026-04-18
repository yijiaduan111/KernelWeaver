"""Code agent wrapper."""

from __future__ import annotations

from .base import BaseAgent
from ..models import AgentContext, PlanProposal, SearchNode, TaskSpec


class CodeAgent(BaseAgent):
    def run(
        self,
        task: TaskSpec,
        node: SearchNode,
        proposal: PlanProposal,
        context: AgentContext,
    ) -> str:
        return self.provider.generate_code(task, node, proposal, context)
