"""Detailed curated Level 1 loss and attention bridge specs."""

from __future__ import annotations

from typing import Any

from ...models import StrategySpec

TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 95): {
        "alias": "KB-L1-P95",
        "task_name": "kernelbench_l1_95_cross_entropy_loss",
        "title": "KernelBench Level 1 / 95 CrossEntropyLoss",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "loss", "cross_entropy", "decomposition"],
        "init_args": [],
        "init_kwargs": {},
        "test_case_specs": [
            {
                "args": [
                    {"kind": "rand", "shape": (128, 64)},
                    {"kind": "randint", "shape": (128,), "low": 0, "high": 64, "dtype": "int64"},
                ]
            },
            {
                "args": [
                    {"kind": "rand", "shape": (192, 96)},
                    {"kind": "randint", "shape": (192,), "low": 0, "high": 96, "dtype": "int64"},
                ]
            },
        ],
        "benchmark_case_specs": [
            {
                "args": [
                    {"kind": "rand", "shape": (256, 128)},
                    {"kind": "randint", "shape": (256,), "low": 0, "high": 128, "dtype": "int64"},
                ]
            },
            {
                "args": [
                    {"kind": "rand", "shape": (384, 192)},
                    {"kind": "randint", "shape": (384,), "low": 0, "high": 192, "dtype": "int64"},
                ]
            },
        ],
        "forward_steps": [
            "logits = predictions\nlabels = targets",
            "output = torch.nn.functional.cross_entropy(logits, labels)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="cross_entropy_logsoftmax_nll",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the CrossEntropyLoss compute step as explicit log_softmax plus nll_loss.",
                instruction="Replace the compute step with log_softmax over dim=1 followed by nll_loss while preserving the logits/labels contract.",
                expected_gain="Expose the loss decomposition clearly for later backend-specific reasoning.",
                good_body="log_probs = torch.nn.functional.log_softmax(logits, dim=1)\noutput = torch.nn.functional.nll_loss(log_probs, labels)\n",
                broken_body="log_probs = torch.nn.functional.log_softmax(logits, dim=0)\noutput = torch.nn.functional.nll_loss(log_probs, labels)\n",
                debug_body="log_probs = torch.nn.functional.log_softmax(logits, dim=1)\noutput = torch.nn.functional.nll_loss(log_probs, labels)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
    (1, 97): {
        "alias": "KB-L1-P97",
        "task_name": "kernelbench_l1_97_scaled_dot_product_attention",
        "title": "KernelBench Level 1 / 97 Scaled Dot Product Attention",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "attention", "softmax", "matmul", "layout_sensitive"],
        "init_args": [],
        "init_kwargs": {},
        "test_case_specs": [
            {
                "args": [
                    {"kind": "rand", "shape": (1, 4, 32, 64)},
                    {"kind": "rand", "shape": (1, 4, 32, 64)},
                    {"kind": "rand", "shape": (1, 4, 32, 64)},
                ]
            },
            {
                "args": [
                    {"kind": "rand", "shape": (2, 4, 48, 64)},
                    {"kind": "rand", "shape": (2, 4, 48, 64)},
                    {"kind": "rand", "shape": (2, 4, 48, 64)},
                ]
            },
        ],
        "benchmark_case_specs": [
            {
                "args": [
                    {"kind": "rand", "shape": (2, 8, 64, 64)},
                    {"kind": "rand", "shape": (2, 8, 64, 64)},
                    {"kind": "rand", "shape": (2, 8, 64, 64)},
                ]
            },
            {
                "args": [
                    {"kind": "rand", "shape": (2, 8, 96, 64)},
                    {"kind": "rand", "shape": (2, 8, 96, 64)},
                    {"kind": "rand", "shape": (2, 8, 96, 64)},
                ]
            },
        ],
        "forward_steps": [
            "query = Q\nkey = K\nvalue = V",
            "output = torch.nn.functional.scaled_dot_product_attention(query, key, value)",
            "return output",
        ],
        "strategies": [
            StrategySpec(
                name="scaled_dot_product_attention_explicit",
                anchor_name="forward_step_2",
                strategy_summary="Rewrite the attention compute step as explicit score, softmax, and value-projection substeps.",
                instruction="Replace the compute step with explicit score computation, scaling, softmax over the last dimension, and final value projection while preserving the Q/K/V roles.",
                expected_gain="Expose the attention dataflow clearly for later Triton or CUDA lowering.",
                good_body="scores = torch.matmul(query, key.transpose(-2, -1))\nscale = 1.0 / (query.size(-1) ** 0.5)\nattn_weights = torch.softmax(scores * scale, dim=-1)\noutput = torch.matmul(attn_weights, value)\n",
                broken_body="scores = torch.matmul(query, query.transpose(-2, -1))\nscale = 1.0 / (query.size(-1) ** 0.5)\nattn_weights = torch.softmax(scores * scale, dim=-1)\noutput = torch.matmul(attn_weights, value)\n",
                debug_body="scores = torch.matmul(query, key.transpose(-2, -1))\nscale = 1.0 / (query.size(-1) ** 0.5)\nattn_weights = torch.softmax(scores * scale, dim=-1)\noutput = torch.matmul(attn_weights, value)\n",
                broken_failure_type="correctness_error",
            ),
        ],
    },
}
