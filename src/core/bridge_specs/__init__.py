"""Curated KernelBench bridge specs grouped by task family.

bridge.py keeps the loading and scaffold assembly logic.
This package only stores hand-curated task metadata and backend tags.
"""

from __future__ import annotations

from typing import Any

from .extra_targets import TARGETS as EXTRA_TARGETS
from .l1_conv import TARGETS as L1_CONV_TARGETS
from .l1_elementwise import TARGETS as L1_ELEMENTWISE_TARGETS
from .l1_loss_attention import TARGETS as L1_LOSS_ATTENTION_TARGETS
from .l1_matmul import TARGETS as L1_MATMUL_TARGETS
from .l1_norm import TARGETS as L1_NORM_TARGETS
from .l1_pooling import TARGETS as L1_POOLING_TARGETS
from .l1_reduction import TARGETS as L1_REDUCTION_TARGETS

TargetRegistry = dict[tuple[int, int], dict[str, Any]]


def _merge_target_groups(*groups: TargetRegistry) -> TargetRegistry:
    merged: TargetRegistry = {}
    for group in groups:
        overlap = set(merged).intersection(group)
        if overlap:
            overlap_list = ', '.join(f'L{level}P{problem_id}' for level, problem_id in sorted(overlap))
            raise ValueError(f'Duplicate curated bridge targets: {overlap_list}')
        merged.update(group)
    return merged


SELECTED_TARGETS: TargetRegistry = _merge_target_groups(
    L1_MATMUL_TARGETS,
    L1_ELEMENTWISE_TARGETS,
    L1_NORM_TARGETS,
    L1_POOLING_TARGETS,
    L1_REDUCTION_TARGETS,
    L1_CONV_TARGETS,
    L1_LOSS_ATTENTION_TARGETS,
    EXTRA_TARGETS,
)

CUDA_ENABLED_TARGETS: set[tuple[int, int]] = set(SELECTED_TARGETS.keys())

NATIVE_CUDA_TARGETS: set[tuple[int, int]] = {
    (1, 25),
    (1, 47),
}

SAFE_CUDA_FORWARD_ONLY_TARGETS: set[tuple[int, int]] = {
    (1, 42),
    (1, 45),
    (1, 50),
    (1, 61),
    (1, 82),
    (1, 89),
    (1, 95),
    (1, 97),
}


def selected_kernelbench_targets() -> list[dict[str, Any]]:
    rows = []
    for (level, problem_id), payload in sorted(SELECTED_TARGETS.items()):
        rows.append(
            {
                'alias': payload.get('alias'),
                'level': level,
                'problem_id': problem_id,
                'task_name': payload['task_name'],
                'title': payload['title'],
                'tags': list(payload['tags']),
            }
        )
    return rows


__all__ = [
    'CUDA_ENABLED_TARGETS',
    'NATIVE_CUDA_TARGETS',
    'SAFE_CUDA_FORWARD_ONLY_TARGETS',
    'SELECTED_TARGETS',
    'selected_kernelbench_targets',
]
