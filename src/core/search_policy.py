"""Small, hardware-neutral search-policy space for bounded harness evolution.

Only observed scalar evidence crosses this interface: no source code, task IDs,
Wiki content, future children, arbitrary Python or evaluator modifications.
TreeMemory remains responsible for pruning, root limits and debug limits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class SearchPolicy:
    schema_version: int = 1
    name: str = "balanced_v0"
    breadth_steps: int = 1
    novelty_weight: float = 0.3
    debug_bonus: float = 0.3
    failed_child_penalty: float = 0.3

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported policy schema")
        if not isinstance(self.name, str) or not self.name or len(self.name) > 80:
            raise ValueError("Invalid policy name")
        if type(self.breadth_steps) is not int or not 0 <= self.breadth_steps <= 3:
            raise ValueError("breadth_steps must be an integer in [0, 3]")
        for key in ("novelty_weight", "debug_bonus", "failed_child_penalty"):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and in [0, 1]")

    def payload(self):
        return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()


def load_policy(path: str | Path) -> SearchPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(SearchPolicy)}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Policy must contain only the declared scalar fields")
    return SearchPolicy(**payload)


@dataclass(frozen=True)
class NodeObservation:
    node_id: str
    is_root: bool
    depth: int
    selected_count: int
    correct: bool
    is_failure: bool
    relative_value: float
    failed_child_fraction: float


def observe_tree(tree, eligible: list[str]) -> tuple[NodeObservation, ...]:
    root_score = tree.nodes[tree.root_id].score
    observations = []
    for node_id in eligible:
        node = tree.nodes[node_id]
        value = 0.0
        if node.correct and math.isfinite(node.score) and node.score > 0 and math.isfinite(root_score) and root_score > 0:
            # Dimensionless within-device evidence, not raw latency across GPUs.
            value = math.log(max(0.01, min(100.0, root_score / node.score)))
        children = [tree.nodes[child] for child in node.child_ids]
        observations.append(NodeObservation(
            node_id, node_id == tree.root_id, node.depth, node.selected_count,
            node.correct, node.is_failure, value,
            sum(child.is_failure for child in children) / len(children) if children else 0.0,
        ))
    return tuple(observations)


def select_parent(policy: SearchPolicy, observations: tuple[NodeObservation, ...]) -> str:
    if not observations:
        raise ValueError("Policy requires eligible observations")
    root = next((node for node in observations if node.is_root), None)
    if root and root.selected_count < policy.breadth_steps:
        return root.node_id
    selections = sum(node.selected_count for node in observations)

    def priority(node):
        utility = node.relative_value if node.correct else -0.5
        if node.is_failure:
            utility += policy.debug_bonus
        bonus = policy.novelty_weight * math.sqrt(math.log(2 + selections) / (1 + node.selected_count))
        score = utility + bonus - policy.failed_child_penalty * node.failed_child_fraction
        return (-score, node.selected_count, node.depth, node.node_id)

    return min(observations, key=priority).node_id
