from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core.execution_facts import ExecutionFacts
from .feedback.schema import FeedbackState
from .deliberation.schema import StrategyPortfolio
from .semantics.schema import SemanticProfile


@dataclass
class TestCase:
    label: str
    args: list[Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    init_args: list[Any] = field(default_factory=list)
    init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySpec:
    name: str
    anchor_name: str
    strategy_summary: str
    instruction: str
    expected_gain: str
    good_body: str
    broken_body: str | None = None
    debug_body: str | None = None
    broken_failure_type: str | None = None


@dataclass
class GroundedRegion:
    anchor_name: str
    region_role: str
    start_line: int
    end_line: int
    source_excerpt: str
    source_hash: str


@dataclass
class TaskSpec:
    name: str
    description: str
    source_code: str
    reference_code: str
    function_name: str
    reference_function_name: str
    test_cases: list[TestCase]
    benchmark_cases: list[TestCase]
    tags: list[str] = field(default_factory=list)
    strategy_catalog: list[StrategySpec] = field(default_factory=list)
    source_origin: str | None = None
    benchmark_family: str | None = None
    entry_kind: str = "callable"
    level: int | None = None
    problem_id: int | None = None
    backend: str | None = None
    source_root: str | None = None
    grounded_regions: list[GroundedRegion] = field(default_factory=list)
    execution_facts: ExecutionFacts | None = None
    semantic_profile: SemanticProfile | None = None
    strategy_portfolio: StrategyPortfolio | None = None

    def strategy_map(self) -> dict[str, StrategySpec]:
        return {strategy.name: strategy for strategy in self.strategy_catalog}


@dataclass
class AnchorEdit:
    anchor_name: str
    instruction: str
    operation: str = "replace"


@dataclass
class PlanProposal:
    strategy_name: str
    strategy_summary: str
    anchor_edits: list[AnchorEdit]
    expected_gain: str
    risk_notes: str = ""


@dataclass
class EvaluationResult:
    compile_ok: bool
    correct: bool
    runtime: float | None
    score: float
    logs: list[str] = field(default_factory=list)
    failure_type: str | None = None
    failure_stage: str = "none"
    reference_runtime: float | None = None
    speedup: float | None = None
    reference_runtimes: dict[str, float | None] = field(default_factory=dict)
    speedups: dict[str, float | None] = field(default_factory=dict)
    primary_reference: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.failure_stage != "none"


@dataclass
class SearchNode:
    node_id: str
    parent_id: str | None
    depth: int
    code: str
    origin: str
    child_ids: list[str] = field(default_factory=list)
    selected_count: int = field(default=0)
    plan_strategy_name: str | None = None
    plan_summary: str | None = None
    anchor_edits: list[AnchorEdit] = field(default_factory=list)
    compile_ok: bool = False
    correct: bool = False
    runtime: float | None = None
    score: float = float("inf")
    logs: list[str] = field(default_factory=list)
    failure_type: str | None = None
    node_status: str = "correct"
    selection_reason: str | None = None
    prune_reason: str | None = None
    debug_attempts: int = 0
    latest_failure_stage: str | None = None
    reference_runtime: float | None = None
    speedup: float | None = None
    reference_runtimes: dict[str, float | None] = field(default_factory=dict)
    speedups: dict[str, float | None] = field(default_factory=dict)
    primary_reference: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.node_status in {"compile_fail", "runtime_fail", "correctness_fail"}

    @property
    def status(self) -> str:
        return self.node_status


@dataclass
class NodeSnapshot:
    node_id: str
    parent_id: str | None
    depth: int
    score: float | None
    status: str
    plan_strategy_name: str | None
    failure_type: str | None
    child_count: int
    origin: str
    selected_count: int
    runtime: float | None
    latest_failure_stage: str | None
    reference_runtime: float | None = None
    speedup: float | None = None
    delta_vs_root: float | None = None
    delta_vs_parent: float | None = None
    failure_log_excerpt: str | None = None
    code_hash: str | None = None


@dataclass
class AgentContext:
    role: str
    current: NodeSnapshot
    root: NodeSnapshot
    related: list[NodeSnapshot] = field(default_factory=list)
    leaders: list[NodeSnapshot] = field(default_factory=list)
    failure: NodeSnapshot | None = None
    strategy_history: list[dict] = field(default_factory=list)
    feedback_state: FeedbackState | None = None


@dataclass
class StarkConfig:
    max_attempts: int = 6
    epsilon: float = 0.4
    root_child_limit: int = 2
    dead_branch_threshold: int = 2
    context_limit: int = 5
    leaderboard_size: int = 3
    debug_retry_limit: int = 1
    benchmark_loops: int = 50
    warmup_loops: int = 5
    seed: int = 7
    verbose: bool = False
    run_profile: str | None = None
    search_profile: str | None = None
    evaluator_profile: str | None = None
    measurement_profile: str | None = None
    provider_name: str | None = None
    agent_provider_profile: str | None = None
    plan_provider: str | None = None
    code_provider: str | None = None
    debug_provider: str | None = None
    search_provider: str | None = None
    preset: str = "default"
    evaluation_profile: str = "kernelbench_reduced_v1"
    kernelbench_evaluator: str = "paper"
    num_correct_trials: int = 1
    num_perf_trials: int = 20
    paper_num_warmup: int = 5
    paper_discard_first: int = 1
    timing_method: str = "cuda_event"
    reference_modes: list[str] = field(default_factory=lambda: ["torch_eager"])
    semantics_enabled: bool = True
    semantics_mode: str = "rule"
    semantics_max_anchor_hints: int = 6
    deliberation_enabled: bool = False
    deliberation_profile: str | None = None
    deliberation_mode: str = "multi_model_v0"
    deliberation_providers: list[str] = field(default_factory=list)
    deliberation_max_strategies: int = 10
    deliberation_strategies_per_model: int = 4
    deliberation_proposal_temperature: float = 0.4
    deliberation_review_temperature: float = 0.1
    evaluator_isolation: str = "off"
    evaluator_timeout_seconds: int = 900


@dataclass
class RunResult:
    task_name: str
    config: StarkConfig
    best_node_id: str
    leaderboard: list[str]
    nodes: dict[str, SearchNode]
    selection_history: list[str]
    stats: dict[str, Any]
    leaderboard_history: list[list[str]] = field(default_factory=list)
    selection_reasons: list[str] = field(default_factory=list)
    selection_exclusions: list[dict[str, str]] = field(default_factory=list)
    pruned_nodes: dict[str, str] = field(default_factory=dict)
    debug_stats: dict[str, Any] = field(default_factory=dict)
    benchmark_family: str | None = None
    level: int | None = None
    problem_id: int | None = None
    backend: str | None = None
    source_origin: str | None = None
    source_root: str | None = None
    workflow: str = "stark"
    run_profile: str | None = None
    search_profile: str | None = None
    evaluator_profile: str | None = None
    measurement_profile: str | None = None
    preset: str = "default"
    evaluation_profile: str = "kernelbench_reduced_v1"
    kernelbench_evaluator: str = "paper"
    grounded_regions: list[GroundedRegion] = field(default_factory=list)
    reference_runtimes: dict[str, float | None] = field(default_factory=dict)
    speedups: dict[str, float | None] = field(default_factory=dict)
    primary_reference: str | None = None
    semantic_profile: SemanticProfile | None = None
    strategy_portfolio: StrategyPortfolio | None = None
    feedback_state: FeedbackState | None = None
