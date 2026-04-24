"""Detailed curated Level 1 reduction bridge specs."""

from __future__ import annotations

from typing import Any

from ...models import StrategySpec

TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 47): {
        "alias": "KB-L1-P47",
        "task_name": "kernelbench_l1_47_sum_reduction",
        "title": "KernelBench Level 1 / 47 Sum reduction over a dimension",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "reduction", "keepdim", "shape_sensitive"],
        "init_args": [1],
        "init_kwargs": {},
        "test_shapes": [(8, 256, 255), (12, 384, 383)],
        "benchmark_shapes": [(16, 512, 511), (20, 640, 639)],
        "input_kind": "rand",
        "forward_steps": [
            "input_tensor = x\nreduction_dim = self.dim",
            "output = torch.sum(input_tensor, dim=reduction_dim, keepdim=True)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="sum_reduction_input_prepare",
                anchor_name="forward_step_1",
                strategy_summary="Prepare the reduction input explicitly before the sum compute step.",
                instruction="Replace the input binding step with a layout-aware preparation step, such as a contiguous copy, while preserving the reduction dimension.",
                expected_gain="Make reduction input preparation explicit before introducing a custom kernel.",
                good_body="input_tensor = x.contiguous()\nreduction_dim = self.dim\n",
                broken_body="input_tensor = x\nreduction_dim = 0\n",
                debug_body="input_tensor = x.contiguous()\nreduction_dim = self.dim\n",
                broken_failure_type="correctness_error",
            ),
            StrategySpec(
                name="sum_reduction_keepdim_explicit",
                anchor_name="forward_step_2",
                strategy_summary="Keep the sum reduction and keepdim contract isolated in one compute step.",
                instruction="Replace the compute step with the exact torch.sum call over reduction_dim and preserve keepdim=True.",
                expected_gain="Keep the reduction boundary explicit while preserving output shape semantics.",
                good_body="output = torch.sum(input_tensor, dim=reduction_dim, keepdim=True)\n",
                broken_body="output = torch.sum(input_tensor, dim=reduction_dim)\n",
                debug_body="output = torch.sum(input_tensor, dim=reduction_dim, keepdim=True)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
    (1, 89): {
        "alias": "KB-L1-P89",
        "task_name": "kernelbench_l1_89_cumsum",
        "title": "KernelBench Level 1 / 89 Cumsum",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "reduction", "scan", "prefix"],
        "init_args": [1],
        "init_kwargs": {},
        "test_shapes": [(128, 128), (160, 160)],
        "benchmark_shapes": [(256, 256), (320, 320)],
        "input_kind": "rand",
        "forward_steps": [
            "input_tensor = x\nscan_dim = self.dim",
            "output = torch.cumsum(input_tensor, dim=scan_dim)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="cumsum_input_prepare",
                anchor_name="forward_step_1",
                strategy_summary="Prepare the scan input explicitly before the cumulative sum step.",
                instruction="Replace the input binding step with a layout-aware preparation step, such as a contiguous copy, while preserving the scan dimension.",
                expected_gain="Make scan input preparation explicit before introducing a custom prefix-scan kernel.",
                good_body="input_tensor = x.contiguous()\nscan_dim = self.dim\n",
                broken_body="input_tensor = x\nscan_dim = 0\n",
                debug_body="input_tensor = x.contiguous()\nscan_dim = self.dim\n",
                broken_failure_type="correctness_error",
            ),
            StrategySpec(
                name="cumsum_compute_explicit",
                anchor_name="forward_step_2",
                strategy_summary="Keep the cumulative sum isolated in one grounded compute step.",
                instruction="Replace the compute step with the exact torch.cumsum call over scan_dim and keep the output variable stable.",
                expected_gain="Keep the scan boundary explicit while preserving prefix order semantics.",
                good_body="output = torch.cumsum(input_tensor, dim=scan_dim)\n",
                broken_body="output = torch.cumsum(input_tensor, dim=0)\n",
                debug_body="output = torch.cumsum(input_tensor, dim=scan_dim)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
}
