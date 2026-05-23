"""Route plan, code, debug, and search roles to different providers."""

from __future__ import annotations

from ..models import AgentContext, PlanProposal, SearchNode, TaskSpec
from .base_provider import AgentProvider


class RoleRoutedProvider(AgentProvider):
    """按 agent 角色把请求路由到不同 provider。"""

    name = "role-routed"

    def __init__(
        self,
        plan_provider: AgentProvider,
        code_provider: AgentProvider,
        debug_provider: AgentProvider,
        search_provider: AgentProvider | None = None,
    ) -> None:
        self.plan_provider = plan_provider
        self.code_provider = code_provider
        self.debug_provider = debug_provider
        self.search_provider = search_provider or code_provider

    def propose_plan(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> PlanProposal:
        return self.plan_provider.propose_plan(task, node, context)

    def generate_code(
        self,
        task: TaskSpec,
        node: SearchNode,
        proposal: PlanProposal,
        context: AgentContext,
    ) -> str:
        return self.code_provider.generate_code(task, node, proposal, context)

    def debug_code(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> str:
        return self.debug_provider.debug_code(task, node, context)

    def generate_search_candidate(
        self,
        task: TaskSpec,
        node: SearchNode,
        context: AgentContext,
    ) -> tuple[PlanProposal, str]:
        return self.search_provider.generate_search_candidate(task, node, context)

    def generate_text(
        self,
        system_prompt: str,
        user_payload: dict,
        temperature: float = 0.2,
        purpose: str = "generic",
    ) -> str:
        return self.plan_provider.generate_text(system_prompt, user_payload, temperature, purpose)

    def close(self) -> None:
        seen: set[int] = set()
        for child in (self.plan_provider, self.code_provider, self.debug_provider, self.search_provider):
            if child is None or id(child) in seen:
                continue
            seen.add(id(child))
            close_fn = getattr(child, 'close', None)
            if callable(close_fn):
                close_fn()
