"""Provider interface shared by all model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import AgentContext, PlanProposal, SearchNode, TaskSpec


class AgentProvider(ABC):
    """Abstract LLM backend contract used by all workflow modes."""

    name = "provider"

    @abstractmethod
    def propose_plan(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> PlanProposal:
        raise NotImplementedError

    @abstractmethod
    def generate_code(
        self,
        task: TaskSpec,
        node: SearchNode,
        proposal: PlanProposal,
        context: AgentContext,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def debug_code(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> str:
        raise NotImplementedError

    def generate_search_candidate(
        self,
        task: TaskSpec,
        node: SearchNode,
        context: AgentContext,
    ) -> tuple[PlanProposal, str]:
        """Fallback single-agent search step.

        Providers can override this to emulate the paper's Search Agent
        more faithfully. The default fallback reuses the existing
        plan/code contract so current providers remain compatible.
        """
        proposal = self.propose_plan(task, node, context)
        code = self.generate_code(task, node, proposal, context)
        return proposal, code

    def generate_text(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.2,
        purpose: str = "generic",
    ) -> str:
        """Generic text-generation hook used by non-agent modules."""
        del system_prompt, user_payload, temperature, purpose
        raise NotImplementedError(f"{self.name} does not support generic text generation.")
