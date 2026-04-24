"""Detailed curated Level 1 pooling bridge specs."""

from __future__ import annotations

from typing import Any

from ...models import StrategySpec

TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 42): {
        "alias": "KB-L1-P42",
        "task_name": "kernelbench_l1_42_maxpool2d",
        "title": "KernelBench Level 1 / 42 Max Pooling 2D",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "pooling", "maxpool", "windowed", "stateful_module"],
        "init_args": [4, 1, 1, 1],
        "init_kwargs": {},
        "test_shapes": [(2, 64, 64, 64), (3, 64, 80, 80)],
        "benchmark_shapes": [(4, 64, 96, 96), (6, 64, 128, 128)],
        "input_kind": "rand",
        "forward_steps": [
            "pool_layer = self.maxpool",
            "output = pool_layer(x)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="maxpool2d_functional_rewrite",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the MaxPool2d compute step as an explicit functional max_pool2d call.",
                instruction="Replace the compute step with torch.nn.functional.max_pool2d using the stored pool_layer state explicitly.",
                expected_gain="Expose the pooling window configuration clearly while preserving MaxPool2d semantics.",
                good_body="output = torch.nn.functional.max_pool2d(x, kernel_size=pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation, ceil_mode=pool_layer.ceil_mode, return_indices=pool_layer.return_indices)\n",
                broken_body="output = torch.nn.functional.max_pool2d(x, kernel_size=pool_layer.kernel_size, stride=pool_layer.stride)\n",
                debug_body="output = torch.nn.functional.max_pool2d(x, kernel_size=pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation, ceil_mode=pool_layer.ceil_mode, return_indices=pool_layer.return_indices)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
    (1, 45): {
        "alias": "KB-L1-P45",
        "task_name": "kernelbench_l1_45_avgpool2d",
        "title": "KernelBench Level 1 / 45 Average Pooling 2D",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "pooling", "avgpool", "windowed", "stateful_module"],
        "init_args": [11],
        "init_kwargs": {},
        "test_shapes": [(1, 64, 96, 96), (2, 64, 128, 128)],
        "benchmark_shapes": [(2, 64, 160, 160), (3, 64, 192, 192)],
        "input_kind": "rand",
        "forward_steps": [
            "pool_layer = self.avg_pool",
            "output = pool_layer(x)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="avgpool2d_functional_rewrite",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the AvgPool2d compute step as an explicit functional avg_pool2d call.",
                instruction="Replace the compute step with torch.nn.functional.avg_pool2d using the stored pool_layer state explicitly.",
                expected_gain="Expose the pooling window configuration clearly while preserving AvgPool2d semantics.",
                good_body="output = torch.nn.functional.avg_pool2d(x, kernel_size=pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, ceil_mode=pool_layer.ceil_mode, count_include_pad=pool_layer.count_include_pad, divisor_override=pool_layer.divisor_override)\n",
                broken_body="output = torch.nn.functional.avg_pool2d(x, kernel_size=pool_layer.kernel_size)\n",
                debug_body="output = torch.nn.functional.avg_pool2d(x, kernel_size=pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, ceil_mode=pool_layer.ceil_mode, count_include_pad=pool_layer.count_include_pad, divisor_override=pool_layer.divisor_override)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
}
