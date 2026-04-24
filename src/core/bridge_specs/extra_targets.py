"""Additional curated bridge specs kept for compatibility and paper-style runs."""

from __future__ import annotations

from typing import Any

from ...models import StrategySpec

TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 31): {
        "alias": "KB-T2",
        "task_name": "kernelbench_l1_31_elu",
        "title": "KernelBench Level 1 / 31 ELU",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "parameterized"],
        "init_args": [1.0],
        "init_kwargs": {},
        "test_shapes": [(32, 256), (48, 384)],
        "benchmark_shapes": [(64, 768), (96, 1024)],
        "input_kind": "symmetric",
        "strategies": [
            StrategySpec(
                name="elu_forward_where",
                anchor_name="forward_body",
                strategy_summary="Rewrite ELU with torch.where using the anchored alpha state.",
                instruction="Replace the forward body with a torch.where formulation that uses self.alpha for the negative branch.",
                expected_gain="Make the activation path explicit for later low-level optimization.",
                good_body="positive = x >= 0\nnegative = self.alpha * (torch.exp(x) - 1.0)\nreturn torch.where(positive, x, negative)\n",
                broken_body="return torch.where(x >= 0, x, torch.exp(x) - 1.0)\n",
                debug_body="positive = x >= 0\nnegative = self.alpha * (torch.exp(x) - 1.0)\nreturn torch.where(positive, x, negative)\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
    (2, 1): {
    "alias": "KB-T12",
    "task_name": "kernelbench_l2_1_conv2d_relu_biasadd",
    "title": "KernelBench Level 2 / 1 Conv2D ReLU BiasAdd",
    "tags": ["kernelbench", "official", "gpu", "triton", "level2", "fusion", "conv"],
    "init_args": [64, 128, 3, (128, 1, 1)],
    "init_kwargs": {},
    "test_shapes": [(2, 64, 32, 32), (3, 64, 40, 40)],
    "benchmark_shapes": [(4, 64, 64, 64), (6, 64, 80, 80)],
    "input_kind": "rand",
    "strategies": [
        StrategySpec(
            name="conv_relu_biasadd_explicit",
            anchor_name="forward_body",
            strategy_summary="Keep the Conv2D path explicit with convolution, ReLU, and bias add in the original order.",
            instruction="Replace the forward body with an explicit conv call, a ReLU activation, and the final bias add using self.bias.",
            expected_gain="Expose the conv-plus-activation-plus-bias fusion boundary without changing module ownership.",
            good_body='conv = self.conv(x)\nactivated = torch.relu(conv)\nreturn activated + self.bias\n',
            broken_body='conv = self.conv(x)\nbiased = conv + self.bias\nreturn torch.relu(biased)\n',
            debug_body='conv = self.conv(x)\nactivated = torch.relu(conv)\nreturn activated + self.bias\n',
            broken_failure_type="correctness_error",
        )
    ],
},
(2, 8): {
    "alias": "KB-T13",
    "task_name": "kernelbench_l2_8_conv3d_globalpool_biasadd_sum",
    "title": "KernelBench Level 2 / 8 Conv3D Divide Max GlobalAvgPool BiasAdd Sum",
    "tags": ["kernelbench", "official", "gpu", "triton", "level2", "fusion", "conv3d", "reduction"],
    "init_args": [8, 16, (3, 3, 3), 2.0, (2, 2, 2), (16, 1, 1, 1), 1],
    "init_kwargs": {},
    "test_shapes": [(2, 8, 8, 32, 32), (3, 8, 10, 40, 40)],
    "benchmark_shapes": [(4, 8, 12, 48, 48), (6, 8, 16, 64, 64)],
    "input_kind": "rand",
    "strategies": [
        StrategySpec(
            name="conv3d_pool_reduce_explicit",
            anchor_name="forward_body",
            strategy_summary="Keep the Conv3D pipeline explicit with divide, max-pool, global average pool, bias add, and final reduction.",
            instruction="Replace the forward body with explicit temporaries for conv, divide, max-pool, global average pool, bias add, and torch.sum over self.sum_dim.",
            expected_gain="Expose the full post-conv pipeline clearly for later low-level optimization or fusion.",
            good_body='conv = self.conv(x)\nscaled = conv / self.divisor\npooled = self.max_pool(scaled)\nglobal_pooled = self.global_avg_pool(pooled)\nshifted = global_pooled + self.bias\nreturn torch.sum(shifted, dim=self.sum_dim)\n',
            broken_body='conv = self.conv(x)\nscaled = conv / self.divisor\npooled = self.max_pool(scaled)\nglobal_pooled = self.global_avg_pool(pooled)\nreturn torch.sum(global_pooled, dim=self.sum_dim)\n',
            debug_body='conv = self.conv(x)\nscaled = conv / self.divisor\npooled = self.max_pool(scaled)\nglobal_pooled = self.global_avg_pool(pooled)\nshifted = global_pooled + self.bias\nreturn torch.sum(shifted, dim=self.sum_dim)\n',
            broken_failure_type="correctness_error",
        )
    ],
},
    (2, 12): {
        "alias": "KB-T3",
        "task_name": "kernelbench_l2_12_gemm_multiply_leakyrelu",
        "title": "KernelBench Level 2 / 12 Gemm Multiply LeakyReLU",
        "tags": ["kernelbench", "official", "gpu", "triton", "level2", "fusion", "gemm"],
        "init_args": [256, 256, 2.0, 0.1],
        "init_kwargs": {},
        "test_shapes": [(16, 256), (24, 256)],
        "benchmark_shapes": [(32, 256), (48, 256)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="gemm_scale_leakyrelu_fused_forward",
                anchor_name="forward_body",
                strategy_summary="Rewrite the fused forward path with explicit linear, scaling, and leaky ReLU operations.",
                instruction="Replace the forward body with an explicit linear call, multiply by self.multiplier, and apply leaky ReLU using the stored slope.",
                expected_gain="Make the fusion boundary explicit for later Triton or fused-kernel optimization.",
                good_body="linear = torch.nn.functional.linear(x, self.gemm.weight, self.gemm.bias)\nscaled = linear * self.multiplier\nreturn torch.nn.functional.leaky_relu(scaled, negative_slope=self.leaky_relu.negative_slope)\n",
                broken_body="linear = torch.nn.functional.linear(x, self.gemm.weight, self.gemm.bias)\nreturn torch.relu(linear)\n",
                debug_body="linear = torch.nn.functional.linear(x, self.gemm.weight, self.gemm.bias)\nscaled = linear * self.multiplier\nreturn torch.nn.functional.leaky_relu(scaled, negative_slope=self.leaky_relu.negative_slope)\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
    (2, 57): {
        "alias": "KB-T7",
        "task_name": "kernelbench_l2_57_conv2d_relu_hardswish",
        "title": "KernelBench Level 2 / 57 Conv2d ReLU HardSwish",
        "tags": ["kernelbench", "official", "gpu", "triton", "level2", "fusion", "conv"],
        "init_args": [8, 32, 3],
        "init_kwargs": {},
        "test_shapes": [(4, 8, 32, 32), (6, 8, 40, 40)],
        "benchmark_shapes": [(8, 8, 64, 64), (12, 8, 80, 80)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="conv_relu_hardswish_functional",
                anchor_name="forward_body",
                strategy_summary="Keep the convolution module but make the fused post-ops explicit with ReLU and HardSwish.",
                instruction="Replace the forward body with an explicit conv call, a ReLU, and torch.nn.functional.hardswish.",
                expected_gain="Expose the conv-plus-activation fusion boundary without changing module ownership.",
                good_body="conv = self.conv(x)\nrelu_out = torch.relu(conv)\nreturn torch.nn.functional.hardswish(relu_out)\n",
                broken_body="return torch.relu(self.conv(x))\n",
                debug_body="conv = self.conv(x)\nrelu_out = torch.relu(conv)\nreturn torch.nn.functional.hardswish(relu_out)\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
    (2, 76): {
        "alias": "KB-T8",
        "task_name": "kernelbench_l2_76_gemm_add_relu",
        "title": "KernelBench Level 2 / 76 Gemm Add ReLU",
        "tags": ["kernelbench", "official", "gpu", "triton", "level2", "fusion", "gemm"],
        "init_args": [1024, 1024, (1024,)],
        "init_kwargs": {},
        "test_shapes": [(16, 1024), (24, 1024)],
        "benchmark_shapes": [(32, 1024), (48, 1024)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="gemm_bias_relu_explicit",
                anchor_name="forward_body",
                strategy_summary="Rewrite the GEMM path with explicit linear, bias add, and ReLU steps.",
                instruction="Replace the forward body with self.gemm(x), add self.bias, and apply torch.relu.",
                expected_gain="Make the fused GEMM + bias + activation structure explicit for later optimization.",
                good_body="projected = self.gemm(x)\nshifted = projected + self.bias\nreturn torch.relu(shifted)\n",
                broken_body="projected = self.gemm(x)\nreturn torch.relu(projected)\n",
                debug_body="projected = self.gemm(x)\nshifted = projected + self.bias\nreturn torch.relu(shifted)\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
    (3, 1): {
        "alias": "KB-T9",
        "task_name": "kernelbench_l3_1_mlp",
        "title": "KernelBench Level 3 / 1 MLP",
        "tags": ["kernelbench", "official", "gpu", "triton", "level3", "mlp"],
        "init_args": [128, [128, 64], 32],
        "init_kwargs": {},
        "test_shapes": [(8, 128), (12, 128)],
        "benchmark_shapes": [(16, 128), (24, 128)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="mlp_forward_explicit_return",
                anchor_name="forward_step_1",
                strategy_summary="Keep the MLP path explicit by storing the sequential output before the final return step.",
                instruction="Replace the first forward step with a named temporary for self.network(x) while preserving the remaining forward steps.",
                expected_gain="Make the forward chain easier to inspect and debug while preserving the Level 3 scaffold.",
                good_body="_stark_forward_value = self.network(x)\n",
                broken_body="_stark_forward_value = self.network(x + 1.0)\n",
                debug_body="_stark_forward_value = self.network(x)\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
    (3, 8): {
        "alias": "KB-T10",
        "task_name": "kernelbench_l3_8_resnet_basic_block",
        "title": "KernelBench Level 3 / 8 ResNetBasicBlock",
        "tags": ["kernelbench", "official", "gpu", "triton", "level3", "cnn_block", "residual", "conv"],
        "init_args": [8, 8, 1],
        "init_kwargs": {},
        "test_shapes": [(2, 8, 16, 16), (3, 8, 20, 20)],
        "benchmark_shapes": [(4, 8, 32, 32), (6, 8, 40, 40)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="resnet_residual_merge_explicit",
                anchor_name="forward_step_8",
                strategy_summary="Rewrite the residual merge as an explicit out = out + identity step.",
                instruction="Replace the residual merge step with an explicit out = out + identity assignment while preserving the surrounding block structure.",
                expected_gain="Keep the residual block semantics explicit for later backend-focused edits.",
                good_body="out = out + identity\n",
                broken_body="out = out - identity\n",
                debug_body="out = out + identity\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
(3, 11): {
    "alias": "KB-T14",
    "task_name": "kernelbench_l3_11_vgg16",
    "title": "KernelBench Level 3 / 11 VGG16",
    "tags": ["kernelbench", "official", "gpu", "triton", "level3", "cnn", "classifier"],
    "init_args": [1000],
    "init_kwargs": {},
    "test_shapes": [(1, 3, 224, 224), (2, 3, 224, 224)],
    "benchmark_shapes": [(2, 3, 224, 224), (3, 3, 224, 224)],
    "input_kind": "rand",
    "strategies": [
        StrategySpec(
            name="vgg_flatten_step_explicit",
            anchor_name="forward_step_2",
            strategy_summary="Keep the classifier boundary explicit by flattening features with start_dim=1.",
            instruction="Replace the flatten step with torch.flatten(x, 1) before the classifier step.",
            expected_gain="Preserve the VGG classifier interface while keeping the forward chain easy to inspect.",
            good_body='x = torch.flatten(x, 1)\n',
            broken_body='x = torch.flatten(x, 0)\n',
            debug_body='x = torch.flatten(x, 1)\n',
            broken_failure_type="correctness_error",
        )
    ],
},
(3, 21): {
    "alias": "KB-T15",
    "task_name": "kernelbench_l3_21_efficientnet_mbconv",
    "title": "KernelBench Level 3 / 21 EfficientNetMBConv",
    "tags": ["kernelbench", "official", "gpu", "triton", "level3", "cnn_block", "depthwise", "residual"],
    "init_args": [112, 192, 5, 2, 6],
    "init_kwargs": {},
    "test_shapes": [(1, 112, 64, 64), (1, 112, 80, 80)],
    "benchmark_shapes": [(1, 112, 96, 96), (2, 112, 112, 112)],
    "input_kind": "rand",
    "strategies": [
        StrategySpec(
            name="mbconv_residual_explicit",
            anchor_name="forward_step_5",
            strategy_summary="Keep the MBConv residual merge explicit when the residual path is enabled.",
            instruction="Replace the residual step with an explicit x = x + identity assignment guarded by self.use_residual.",
            expected_gain="Preserve the residual structure while making the merge point explicit for later backend-focused edits.",
            good_body='if self.use_residual:\n    x = x + identity\n',
            broken_body='if self.use_residual:\n    x = identity\n',
            debug_body='if self.use_residual:\n    x = x + identity\n',
            broken_failure_type="correctness_error",
        )
    ],
},
    (3, 31): {
        "alias": "KB-T11",
        "task_name": "kernelbench_l3_31_vision_attention",
        "title": "KernelBench Level 3 / 31 VisionAttention",
        "tags": ["kernelbench", "official", "gpu", "triton", "level3", "attention"],
        "init_args": [64, 4],
        "init_kwargs": {},
        "test_shapes": [(1, 64, 8, 8), (2, 64, 8, 8)],
        "benchmark_shapes": [(2, 64, 16, 16), (4, 64, 16, 16)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="attention_residual_norm_explicit",
                anchor_name="forward_step_4",
                strategy_summary="Make the attention residual and LayerNorm step explicit inside one forward step anchor.",
                instruction="Replace the residual normalization step with an explicit residual temporary followed by LayerNorm.",
                expected_gain="Expose the attention residual path clearly without changing the surrounding reshape logic.",
                good_body="residual = attn_output + x\nx = self.norm(residual)\n",
                broken_body="x = self.norm(attn_output)\n",
                debug_body="residual = attn_output + x\nx = self.norm(residual)\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
}
