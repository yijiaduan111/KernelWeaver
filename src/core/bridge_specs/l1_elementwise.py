"""Detailed curated Level 1 elementwise bridge specs."""

from __future__ import annotations

from typing import Any

from ...models import StrategySpec

TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 20): {
        "alias": "KB-L1-P20",
        "task_name": "kernelbench_l1_20_leakyrelu",
        "title": "KernelBench Level 1 / 20 LeakyReLU",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "activation", "elementwise", "parameterized", "branchy"],
        "init_args": [],
        "init_kwargs": {},
        "test_shapes": [(16, 1024), (24, 2048)],
        "benchmark_shapes": [(32, 4096), (48, 6144)],
        "input_kind": "symmetric",
        "forward_steps": [
            "input_tensor = x\nnegative_slope = self.negative_slope",
            "output = torch.nn.functional.leaky_relu(input_tensor, negative_slope=negative_slope)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="leakyrelu_branch_explicit",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite LeakyReLU as an explicit positive/negative branch selection.",
                instruction="Replace the compute step with a torch.where formulation that preserves the stored negative_slope parameter.",
                expected_gain="Expose the branch structure explicitly for later low-level kernel lowering.",
                good_body="output = torch.where(input_tensor >= 0, input_tensor, input_tensor * negative_slope)\n",
                broken_body="output = torch.relu(input_tensor)\n",
                debug_body="output = torch.where(input_tensor >= 0, input_tensor, input_tensor * negative_slope)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
    (1, 25): {
        "alias": "KB-L1-P25",
        "task_name": "kernelbench_l1_25_swish",
        "title": "KernelBench Level 1 / 25 Swish",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "activation", "elementwise", "sigmoid"],
        "init_args": [],
        "init_kwargs": {},
        "test_shapes": [(16, 1024), (24, 2048)],
        "benchmark_shapes": [(32, 4096), (48, 6144)],
        "input_kind": "symmetric",
        "forward_steps": [
            "input_tensor = x",
            "output = input_tensor * torch.sigmoid(input_tensor)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="swish_input_prepare",
                anchor_name="forward_step_1",
                strategy_summary="Prepare the Swish input explicitly before the nonlinear compute step.",
                instruction="Replace the input binding step with a layout-aware preparation step, such as a contiguous copy, while preserving the original tensor values.",
                expected_gain="Make input preparation explicit before introducing a custom CUDA or Triton Swish implementation.",
                good_body="input_tensor = x.contiguous()\n",
                broken_body="input_tensor = x + 1.0\n",
                debug_body="input_tensor = x.contiguous()\n",
                broken_failure_type="correctness_error",
            ),
            StrategySpec(
                name="swish_sigmoid_temp",
                anchor_name="forward_step_2",
                strategy_summary="Materialize the sigmoid temporary explicitly inside the Swish compute step.",
                instruction="Replace the compute step with a two-line sigmoid temporary plus multiply while keeping the output variable name stable.",
                expected_gain="Expose the nonlinear dataflow explicitly for later kernel fusion.",
                good_body="sigmoid_x = torch.sigmoid(input_tensor)\noutput = input_tensor * sigmoid_x\n",
                broken_body="sigmoid_x = torch.sigmoid(input_tensor)\noutput = input_tensor + sigmoid_x\n",
                debug_body="sigmoid_x = torch.sigmoid(input_tensor)\noutput = input_tensor * sigmoid_x\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
}
