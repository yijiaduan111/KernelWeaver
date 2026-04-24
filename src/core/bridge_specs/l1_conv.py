"""Detailed curated Level 1 convolution bridge specs."""

from __future__ import annotations

from typing import Any

from ...models import StrategySpec

TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 50): {
        "alias": "KB-L1-P50",
        "task_name": "kernelbench_l1_50_conv2d_standard",
        "title": "KernelBench Level 1 / 50 Standard Conv2D",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "conv", "standard_conv", "layout_sensitive", "stateful_module"],
        "init_args": [1000],
        "init_kwargs": {},
        "test_shapes": [(2, 3, 64, 64), (3, 3, 96, 96)],
        "benchmark_shapes": [(4, 3, 128, 128), (6, 3, 160, 160)],
        "input_kind": "rand",
        "forward_steps": [
            "conv_layer = self.conv1",
            "output = conv_layer(x)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="conv2d_functional_rewrite",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the Conv2D compute step as an explicit functional conv2d call.",
                instruction="Replace the compute step with torch.nn.functional.conv2d using conv_layer weights, bias, stride, padding, dilation, and groups explicitly.",
                expected_gain="Expose Conv2D state explicitly while preserving the module-owned convolution contract.",
                good_body="output = torch.nn.functional.conv2d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)\n",
                broken_body="output = torch.nn.functional.conv2d(x, conv_layer.weight, None, stride=conv_layer.stride, padding=conv_layer.padding)\n",
                debug_body="output = torch.nn.functional.conv2d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
    (1, 61): {
        "alias": "KB-L1-P61",
        "task_name": "kernelbench_l1_61_conv_transpose3d",
        "title": "KernelBench Level 1 / 61 ConvTranspose3D",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "conv", "conv3d", "transpose_conv", "layout_sensitive", "stateful_module"],
        "init_args": [48, 48, 3],
        "init_kwargs": {},
        "test_shapes": [(1, 48, 8, 8, 8), (1, 48, 10, 10, 10)],
        "benchmark_shapes": [(2, 48, 12, 12, 12), (2, 48, 14, 14, 14)],
        "input_kind": "rand",
        "forward_steps": [
            "conv_layer = self.conv_transpose3d",
            "output = conv_layer(x)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="conv_transpose3d_functional_rewrite",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the ConvTranspose3D compute step as an explicit functional conv_transpose3d call.",
                instruction="Replace the compute step with torch.nn.functional.conv_transpose3d using conv_layer weights, bias, stride, padding, output_padding, groups, and dilation explicitly.",
                expected_gain="Expose transpose-convolution state explicitly while preserving the module-owned contract.",
                good_body="output = torch.nn.functional.conv_transpose3d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding, output_padding=conv_layer.output_padding, groups=conv_layer.groups, dilation=conv_layer.dilation)\n",
                broken_body="output = torch.nn.functional.conv_transpose3d(x, conv_layer.weight, None, stride=conv_layer.stride, padding=conv_layer.padding)\n",
                debug_body="output = torch.nn.functional.conv_transpose3d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding, output_padding=conv_layer.output_padding, groups=conv_layer.groups, dilation=conv_layer.dilation)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
    (1, 82): {
        "alias": "KB-L1-P82",
        "task_name": "kernelbench_l1_82_depthwise_conv2d",
        "title": "KernelBench Level 1 / 82 Depthwise Conv2D",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "conv", "depthwise", "layout_sensitive", "stateful_module"],
        "init_args": [64, 3, 1, 0],
        "init_kwargs": {},
        "test_shapes": [(2, 64, 64, 64), (3, 64, 80, 80)],
        "benchmark_shapes": [(4, 64, 96, 96), (6, 64, 128, 128)],
        "input_kind": "rand",
        "forward_steps": [
            "conv_layer = self.conv2d",
            "output = conv_layer(x)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="depthwise_conv2d_functional_rewrite",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the depthwise Conv2D compute step as an explicit functional conv2d call.",
                instruction="Replace the compute step with torch.nn.functional.conv2d using conv_layer weights, bias, stride, padding, dilation, and groups explicitly.",
                expected_gain="Expose depthwise convolution state explicitly while preserving the module-owned contract.",
                good_body="output = torch.nn.functional.conv2d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)\n",
                broken_body="output = torch.nn.functional.conv2d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding)\n",
                debug_body="output = torch.nn.functional.conv2d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
}
