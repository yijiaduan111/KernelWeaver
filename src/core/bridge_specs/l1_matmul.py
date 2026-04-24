"""Detailed curated Level 1 matmul bridge specs."""

from __future__ import annotations

from typing import Any

from ...models import StrategySpec

TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 1): {
        "alias": "KB-L1-P1",
        "task_name": "kernelbench_l1_1_square_matmul",
        "title": "KernelBench Level 1 / 1 Square matrix multiplication",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "matmul", "dense_linear_algebra", "layout_sensitive"],
        "init_args": [],
        "init_kwargs": {},
        "test_case_specs": [
            {
                "args": [
                    {"kind": "rand", "shape": (128, 128)},
                    {"kind": "rand", "shape": (128, 128)},
                ]
            },
            {
                "args": [
                    {"kind": "rand", "shape": (192, 192)},
                    {"kind": "rand", "shape": (192, 192)},
                ]
            },
        ],
        "benchmark_case_specs": [
            {
                "args": [
                    {"kind": "rand", "shape": (256, 256)},
                    {"kind": "rand", "shape": (256, 256)},
                ]
            },
            {
                "args": [
                    {"kind": "rand", "shape": (384, 384)},
                    {"kind": "rand", "shape": (384, 384)},
                ]
            },
        ],
        "forward_steps": [
            "left = A\nright = B",
            "output = torch.matmul(left, right)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="square_matmul_layout_prep",
                anchor_name="forward_step_1",
                strategy_summary="Prepare the two GEMM operands explicitly before the main multiply.",
                instruction="Replace the input binding step with a layout-aware preparation step, such as contiguous copies, while preserving the left/right operand roles.",
                expected_gain="Make operand layout handling explicit without changing the square GEMM contract.",
                good_body="left = A.contiguous()\nright = B.contiguous()\n",
                broken_body="left = A\nright = A\n",
                debug_body="left = A.contiguous()\nright = B.contiguous()\n",
                broken_failure_type="correctness_error",
            ),
            StrategySpec(
                name="square_matmul_compute_call",
                anchor_name="forward_step_2",
                strategy_summary="Keep the square GEMM call isolated in one grounded compute step.",
                instruction="Replace the compute step with the exact torch.matmul call over the prepared operands and keep the output variable stable.",
                expected_gain="Keep the contraction site explicit for later Triton or CUDA lowering.",
                good_body="output = torch.matmul(left, right)\n",
                broken_body="output = torch.matmul(left, left)\n",
                debug_body="output = torch.matmul(left, right)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
    (1, 10): {
        "alias": "KB-L1-P10",
        "task_name": "kernelbench_l1_10_tensor_matmul_3d",
        "title": "KernelBench Level 1 / 10 3D tensor matrix multiplication",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "matmul", "batched_matmul", "layout_sensitive"],
        "init_args": [],
        "init_kwargs": {},
        "test_case_specs": [
            {
                "args": [
                    {"kind": "rand", "shape": (4, 64, 128)},
                    {"kind": "rand", "shape": (128, 96)},
                ]
            },
            {
                "args": [
                    {"kind": "rand", "shape": (6, 96, 160)},
                    {"kind": "rand", "shape": (160, 128)},
                ]
            },
        ],
        "benchmark_case_specs": [
            {
                "args": [
                    {"kind": "rand", "shape": (8, 128, 256)},
                    {"kind": "rand", "shape": (256, 192)},
                ]
            },
            {
                "args": [
                    {"kind": "rand", "shape": (10, 160, 320)},
                    {"kind": "rand", "shape": (320, 256)},
                ]
            },
        ],
        "forward_steps": [
            "left = A\nright = B",
            "output = torch.matmul(left, right)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="batched_matmul_layout_prep",
                anchor_name="forward_step_1",
                strategy_summary="Prepare the batched GEMM operands explicitly before the main multiply.",
                instruction="Replace the input binding step with a layout-aware preparation step, such as contiguous copies, while preserving the tensor/matrix roles.",
                expected_gain="Make batched operand layout handling explicit without changing the contraction dimensions.",
                good_body="left = A.contiguous()\nright = B.contiguous()\n",
                broken_body="left = A\nright = A\n",
                debug_body="left = A.contiguous()\nright = B.contiguous()\n",
                broken_failure_type="correctness_error",
            ),
            StrategySpec(
                name="batched_matmul_compute_call",
                anchor_name="forward_step_2",
                strategy_summary="Keep the tensor-matrix multiply isolated in one grounded compute step.",
                instruction="Replace the compute step with the exact torch.matmul call over the prepared operands and keep the output variable stable.",
                expected_gain="Keep the batched contraction site explicit for later Triton or CUDA lowering.",
                good_body="output = torch.matmul(left, right)\n",
                broken_body="output = torch.matmul(left, left)\n",
                debug_body="output = torch.matmul(left, right)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
}
