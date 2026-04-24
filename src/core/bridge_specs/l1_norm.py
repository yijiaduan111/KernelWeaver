"""Detailed curated Level 1 normalization bridge specs."""

from __future__ import annotations

from typing import Any

from ...models import StrategySpec

TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 33): {
        "alias": "KB-L1-P33",
        "task_name": "kernelbench_l1_33_batchnorm",
        "title": "KernelBench Level 1 / 33 BatchNorm",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "norm", "batchnorm", "stateful_module", "running_stats"],
        "init_args": [64],
        "init_kwargs": {},
        "test_shapes": [(4, 64, 32, 32), (6, 64, 48, 48)],
        "benchmark_shapes": [(8, 64, 64, 64), (12, 64, 80, 80)],
        "input_kind": "rand",
        "forward_steps": [
            "norm_layer = self.bn",
            "output = norm_layer(x)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="batchnorm_functional_rewrite",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the BatchNorm compute step as an explicit functional batch_norm call.",
                instruction="Replace the compute step with torch.nn.functional.batch_norm using norm_layer state and preserve norm_layer.training semantics.",
                expected_gain="Expose BatchNorm statistics and affine state explicitly while preserving module semantics.",
                good_body="output = torch.nn.functional.batch_norm(x, norm_layer.running_mean, norm_layer.running_var, norm_layer.weight, norm_layer.bias, training=norm_layer.training, momentum=norm_layer.momentum, eps=norm_layer.eps)\n",
                broken_body="output = torch.nn.functional.batch_norm(x, norm_layer.running_mean, norm_layer.running_var, norm_layer.weight, norm_layer.bias, training=False, momentum=norm_layer.momentum, eps=norm_layer.eps)\n",
                debug_body="output = torch.nn.functional.batch_norm(x, norm_layer.running_mean, norm_layer.running_var, norm_layer.weight, norm_layer.bias, training=norm_layer.training, momentum=norm_layer.momentum, eps=norm_layer.eps)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
    (1, 40): {
        "alias": "KB-L1-P40",
        "task_name": "kernelbench_l1_40_layernorm",
        "title": "KernelBench Level 1 / 40 LayerNorm",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "norm", "layernorm", "stateful_module"],
        "init_args": [(64, 64, 64)],
        "init_kwargs": {},
        "test_shapes": [(2, 64, 64, 64), (3, 64, 64, 64)],
        "benchmark_shapes": [(4, 64, 64, 64), (6, 64, 64, 64)],
        "input_kind": "rand",
        "forward_steps": [
            "norm_layer = self.ln",
            "output = norm_layer(x)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="layernorm_functional_rewrite",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the LayerNorm compute step as an explicit functional layer_norm call.",
                instruction="Replace the compute step with torch.nn.functional.layer_norm using norm_layer.normalized_shape, weight, bias, and eps.",
                expected_gain="Expose LayerNorm state explicitly while preserving the normalization contract.",
                good_body="output = torch.nn.functional.layer_norm(x, norm_layer.normalized_shape, norm_layer.weight, norm_layer.bias, norm_layer.eps)\n",
                broken_body="output = x - x.mean(dim=-1, keepdim=True)\n",
                debug_body="output = torch.nn.functional.layer_norm(x, norm_layer.normalized_shape, norm_layer.weight, norm_layer.bias, norm_layer.eps)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
}
