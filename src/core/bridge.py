"""Bridge external tasks into STARK's internal task model.

The most important path in this file is the KernelBench bridge. It reads
official benchmark problems from a read-only external clone and converts
them into anchored `TaskSpec` objects that can be consumed by the same
workflow used for demo and Triton tasks.
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import GroundedRegion, StrategySpec, TaskSpec, TestCase


class BridgeLoadError(ValueError):
    pass


@dataclass(slots=True)
class BridgeTaskConfig:
    name: str
    description: str
    source_path: str | Path
    reference_path: str | Path
    function_name: str
    reference_function_name: str
    test_cases: list[TestCase]
    benchmark_cases: list[TestCase]
    tags: list[str] = field(default_factory=list)
    strategy_catalog: list[StrategySpec] = field(default_factory=list)
    source_origin: str | None = None
    benchmark_family: str = "kernelbench"


@dataclass(slots=True)
class OfficialProblemInfo:
    path: Path
    source_code: str
    description: str
    imports_block: str
    class_preamble: str
    init_signature: str
    init_body: str
    forward_signature: str
    forward_body: str
    forward_steps: list[str] = field(default_factory=list)


_SELECTED_TARGETS: dict[tuple[int, int], dict[str, Any]] = {
    (1, 25): {
        "alias": "KB-T1",
        "task_name": "kernelbench_l1_25_swish",
        "title": "KernelBench Level 1 / 25 Swish",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "elementwise"],
        "init_args": [],
        "init_kwargs": {},
        "test_shapes": [(32, 256), (48, 384)],
        "benchmark_shapes": [(64, 768), (96, 1024)],
        "input_kind": "symmetric",
        "strategies": [
            StrategySpec(
                name="swish_forward_rewrite",
                anchor_name="forward_body",
                strategy_summary="Rewrite Swish using an explicit sigmoid temporary inside the anchored forward body.",
                instruction="Replace the forward body with an explicit sigmoid temporary and a final multiply.",
                expected_gain="Make the nonlinear forward path easier to optimize or later fuse.",
                good_body="sigmoid_x = torch.sigmoid(x)\nreturn x * sigmoid_x\n",
                broken_body="return x + torch.sigmoid(x)\n",
                debug_body="sigmoid_x = torch.sigmoid(x)\nreturn x * sigmoid_x\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
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
    (1, 33): {
        "alias": "KB-T4",
        "task_name": "kernelbench_l1_33_batchnorm",
        "title": "KernelBench Level 1 / 33 BatchNorm",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "norm", "batchnorm"],
        "init_args": [32],
        "init_kwargs": {},
        "test_shapes": [(8, 32, 16, 16), (12, 32, 20, 20)],
        "benchmark_shapes": [(16, 32, 32, 32), (24, 32, 40, 40)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="batchnorm_functional_eval",
                anchor_name="forward_body",
                strategy_summary="Rewrite BatchNorm with an explicit functional call in eval mode.",
                instruction="Replace the forward body with torch.nn.functional.batch_norm using the stored BatchNorm state in eval mode.",
                expected_gain="Make the normalization path explicit while preserving parameter and running-stat usage.",
                good_body="return torch.nn.functional.batch_norm(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, training=False, momentum=self.bn.momentum, eps=self.bn.eps)\n",
                broken_body="return torch.nn.functional.batch_norm(x, None, None, self.bn.weight, self.bn.bias, training=True, momentum=self.bn.momentum, eps=self.bn.eps)\n",
                debug_body="return torch.nn.functional.batch_norm(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, training=False, momentum=self.bn.momentum, eps=self.bn.eps)\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
    (1, 40): {
        "alias": "KB-T5",
        "task_name": "kernelbench_l1_40_layernorm",
        "title": "KernelBench Level 1 / 40 LayerNorm",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "norm", "layernorm"],
        "init_args": [(32, 16, 16)],
        "init_kwargs": {},
        "test_shapes": [(4, 32, 16, 16), (6, 32, 16, 16)],
        "benchmark_shapes": [(8, 32, 16, 16), (12, 32, 16, 16)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="layernorm_functional_rewrite",
                anchor_name="forward_body",
                strategy_summary="Rewrite LayerNorm with torch.nn.functional.layer_norm using the stored affine state.",
                instruction="Replace the forward body with torch.nn.functional.layer_norm using self.ln.normalized_shape, self.ln.weight, self.ln.bias, and self.ln.eps.",
                expected_gain="Keep the normalization semantics explicit for later backend-specific lowering.",
                good_body="return torch.nn.functional.layer_norm(x, self.ln.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)\n",
                broken_body="return x - x.mean(dim=-1, keepdim=True)\n",
                debug_body="return torch.nn.functional.layer_norm(x, self.ln.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)\n",
                broken_failure_type="correctness_error",
            )
        ],
    },
    (1, 47): {
        "alias": "KB-T6",
        "task_name": "kernelbench_l1_47_sum_reduction",
        "title": "KernelBench Level 1 / 47 Sum reduction over a dimension",
        "tags": ["kernelbench", "official", "gpu", "triton", "level1", "reduction"],
        "init_args": [1],
        "init_kwargs": {},
        "test_shapes": [(8, 256, 255), (12, 384, 383)],
        "benchmark_shapes": [(16, 512, 511), (20, 640, 639)],
        "input_kind": "rand",
        "strategies": [
            StrategySpec(
                name="sum_keepdim_explicit",
                anchor_name="forward_body",
                strategy_summary="Keep the reduction explicit with torch.sum and the stored reduction dimension.",
                instruction="Replace the forward body with torch.sum over self.dim and keep keepdim=True.",
                expected_gain="Expose the reduction boundary clearly while preserving shape semantics.",
                good_body="return torch.sum(x, dim=self.dim, keepdim=True)\n",
                broken_body="return torch.sum(x, dim=self.dim)\n",
                debug_body="return torch.sum(x, dim=self.dim, keepdim=True)\n",
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

_CUDA_ENABLED_TARGETS: set[tuple[int, int]] = set(_SELECTED_TARGETS.keys())


def _region_role(anchor_name: str) -> str:
    if anchor_name.startswith("forward_step_"):
        return "forward_step"
    return anchor_name


def selected_kernelbench_targets() -> list[dict[str, Any]]:
    """Return the fixed KernelBench target list enabled in the current stage."""
    rows = []
    for (level, problem_id), payload in sorted(_SELECTED_TARGETS.items()):
        rows.append(
            {
                "alias": payload.get("alias"),
                "level": level,
                "problem_id": problem_id,
                "task_name": payload["task_name"],
                "title": payload["title"],
                "tags": list(payload["tags"]),
            }
        )
    return rows


class KernelBenchTaskBridge:
    def load_task(self, config: BridgeTaskConfig) -> TaskSpec:
        """Load a local hand-written callable task into a `TaskSpec`."""
        if not config.test_cases:
            raise BridgeLoadError(f"Bridge task '{config.name}' must define at least one test case.")
        if not config.benchmark_cases:
            raise BridgeLoadError(f"Bridge task '{config.name}' must define at least one benchmark case.")
        source_path = Path(config.source_path)
        reference_path = Path(config.reference_path)
        source_code = self._load_python_source(source_path, config.function_name, label="candidate")
        reference_code = self._load_python_source(reference_path, config.reference_function_name, label="reference")
        return TaskSpec(
            name=config.name,
            description=config.description,
            source_code=source_code,
            reference_code=reference_code,
            function_name=config.function_name,
            reference_function_name=config.reference_function_name,
            test_cases=list(config.test_cases),
            benchmark_cases=list(config.benchmark_cases),
            tags=list(config.tags),
            strategy_catalog=list(config.strategy_catalog),
            source_origin=config.source_origin or str(source_path),
            benchmark_family=config.benchmark_family,
            entry_kind="callable",
        )

    def load_official_problem(
        self,
        kernelbench_root: str | Path,
        level: int,
        problem_id: int,
        backend: str = "triton",
    ) -> TaskSpec:
        """Load one official KernelBench problem and build an anchored scaffold.

        The external benchmark file remains untouched. STARK extracts the
        official `Model` structure, synthesizes a `ModelNew` scaffold, and
        injects grounded edit anchors so the agents can make local changes
        without rewriting the full module arbitrarily.
        """
        if backend not in {"triton", "cuda"}:
            raise BridgeLoadError(f"Unsupported KernelBench backend: {backend}. Supported backends: triton, cuda.")
        target = _SELECTED_TARGETS.get((level, problem_id))
        if target is None:
            supported = ", ".join(f"L{item['level']}/P{item['problem_id']}" for item in selected_kernelbench_targets())
            raise BridgeLoadError(
                f"KernelBench problem L{level}/P{problem_id} is not enabled in this stage. Supported targets: {supported}"
            )
        if backend == "cuda":
            if (level, problem_id) not in _CUDA_ENABLED_TARGETS:
                supported = ", ".join(f"L{item[0]}/P{item[1]}" for item in sorted(_CUDA_ENABLED_TARGETS))
                raise BridgeLoadError(f"CUDA backend is only enabled for {supported} in this stage.")
        root = Path(kernelbench_root)
        problem_path = self._resolve_problem_path(root, level, problem_id)
        info = self._inspect_official_problem(problem_path)
        test_cases = self._build_cases(target, kind="test")
        benchmark_cases = self._build_cases(target, kind="benchmark")
        scaffold = self._build_model_scaffold(info, level=level, backend=backend, level_problem=(level, problem_id))
        grounded_regions = self._extract_grounded_regions(scaffold)
        tags = [backend if tag == "triton" else tag for tag in target["tags"]]
        if backend == "cuda" and "native_cuda" not in tags:
            tags.append("native_cuda")
        strategy_catalog = self._strategy_catalog_for_backend((level, problem_id), backend, target["strategies"])
        return TaskSpec(
            name=target["task_name"],
            description=f"{target['title']} (official KernelBench task with reduced local evaluation profile)",
            source_code=scaffold,
            reference_code=info.source_code,
            function_name="ModelNew",
            reference_function_name="Model",
            test_cases=test_cases,
            benchmark_cases=benchmark_cases,
            tags=tags,
            strategy_catalog=strategy_catalog,
            source_origin=str(problem_path),
            benchmark_family="kernelbench",
            entry_kind="model_class",
            level=level,
            problem_id=problem_id,
            backend=backend,
            source_root=str(root),
            grounded_regions=grounded_regions,
        )

    def _resolve_problem_path(self, kernelbench_root: Path, level: int, problem_id: int) -> Path:
        level_dir = kernelbench_root / "KernelBench" / f"level{level}"
        if not level_dir.exists():
            raise BridgeLoadError(f"KernelBench level directory does not exist: {level_dir}")
        matches = sorted(level_dir.glob(f"{problem_id}_*.py"))
        if not matches:
            raise BridgeLoadError(f"KernelBench problem L{level}/P{problem_id} was not found under {level_dir}")
        return matches[0]

    def _inspect_official_problem(self, path: Path) -> OfficialProblemInfo:
        """Extract the parts of an official benchmark module needed for scaffolding."""
        source = path.read_text(encoding="utf-8")
        try:
            module = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            location = f"{path}:{exc.lineno}:{exc.offset}"
            raise BridgeLoadError(f"Official KernelBench source has invalid syntax at {location}: {exc.msg}") from exc
        model_class = self._find_class(module, "Model", path)
        self._find_function(module, "get_inputs", path)
        self._find_function(module, "get_init_inputs", path)
        init_node = self._find_method(model_class, "__init__", path)
        forward_node = self._find_method(model_class, "forward", path)
        imports_block = self._extract_imports(source, module)
        class_preamble = self._extract_class_preamble(source, model_class)
        init_signature = self._extract_signature(source, init_node)
        forward_signature = self._extract_signature(source, forward_node)
        init_body = self._extract_body(source, init_node, drop_super_init=True)
        forward_body = self._extract_body(source, forward_node, drop_super_init=False)
        forward_steps = self._extract_forward_steps(source, forward_node)
        description = ast.get_docstring(model_class) or path.stem.replace("_", " ")
        return OfficialProblemInfo(
            path=path,
            source_code=source,
            description=description,
            imports_block=imports_block,
            class_preamble=class_preamble,
            init_signature=init_signature,
            init_body=init_body,
            forward_signature=forward_signature,
            forward_body=forward_body,
            forward_steps=forward_steps,
        )

    @staticmethod
    def _find_class(module: ast.Module, class_name: str, path: Path) -> ast.ClassDef:
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        raise BridgeLoadError(f"Class '{class_name}' was not found in {path}")

    @staticmethod
    def _find_function(module: ast.Module, function_name: str, path: Path) -> ast.FunctionDef | ast.AsyncFunctionDef:
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return node
        raise BridgeLoadError(f"Function '{function_name}' was not found in {path}")

    @staticmethod
    def _find_method(class_node: ast.ClassDef, method_name: str, path: Path) -> ast.FunctionDef | ast.AsyncFunctionDef:
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                return node
        raise BridgeLoadError(f"Method '{method_name}' was not found in class '{class_node.name}' from {path}")

    @staticmethod
    def _extract_imports(source: str, module: ast.Module) -> str:
        chunks: list[str] = []
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast.get_source_segment(source, node)
                if segment:
                    chunks.append(segment.strip())
        return "\n".join(chunks)

    @staticmethod
    def _extract_signature(source: str, function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        segment = ast.get_source_segment(source, function_node)
        if not segment:
            raise BridgeLoadError(f"Failed to extract signature for function '{function_node.name}'")
        first_line = segment.splitlines()[0].strip()
        if not first_line.startswith("def ") or not first_line.endswith(":"):
            raise BridgeLoadError(f"Unsupported function header for '{function_node.name}': {first_line}")
        return first_line[4:-1].strip()

    @staticmethod
    def _extract_body(
        source: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
        drop_super_init: bool,
    ) -> str:
        statements = list(function_node.body)
        if statements and isinstance(statements[0], ast.Expr) and isinstance(getattr(statements[0], "value", None), ast.Constant):
            if isinstance(statements[0].value.value, str):
                statements = statements[1:]
        chunks: list[str] = []
        for statement in statements:
            if drop_super_init and _is_super_init_statement(statement):
                continue
            segment = ast.get_source_segment(source, statement)
            if segment:
                chunks.append(segment)
        return textwrap.dedent("\n".join(chunks)).strip("\n")

    @staticmethod
    def _extract_class_preamble(source: str, class_node: ast.ClassDef) -> str:
        chunks: list[str] = []
        for statement in class_node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(statement, ast.Expr) and isinstance(getattr(statement, "value", None), ast.Constant):
                if isinstance(statement.value.value, str):
                    continue
            segment = ast.get_source_segment(source, statement)
            if segment:
                chunks.append(textwrap.dedent(segment).strip("\n"))
        return "\n".join(chunks).strip("\n")

    @staticmethod
    def _extract_forward_steps(
        source: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        """Split a forward body into statement-sized editable steps.

        Level 3 tasks benefit from finer-grained anchors because their
        forward methods are usually multi-stage blocks instead of one
        compact expression.
        """
        statements = list(function_node.body)
        if statements and isinstance(statements[0], ast.Expr) and isinstance(getattr(statements[0], "value", None), ast.Constant):
            if isinstance(statements[0].value.value, str):
                statements = statements[1:]
        steps: list[str] = []
        for statement in statements:
            segment = ast.get_source_segment(source, statement)
            if segment:
                steps.append(textwrap.dedent(segment).strip("\n"))
        if len(steps) == 1 and statements and isinstance(statements[0], ast.Return):
            return KernelBenchTaskBridge._split_single_return_step(source, statements[0])
        return steps

    @staticmethod
    def _split_single_return_step(source: str, statement: ast.Return) -> list[str]:
        """Turn a one-line return into two editable steps for Level 3 scaffolds."""
        if statement.value is None:
            return ["return None"]
        expression = ast.get_source_segment(source, statement.value)
        if not expression:
            return ["return None"]
        temp_name = "_stark_forward_value"
        return [f"{temp_name} = {expression}", f"return {temp_name}"]

    def _build_model_scaffold(
        self,
        info: OfficialProblemInfo,
        level: int,
        backend: str,
        level_problem: tuple[int, int],
    ) -> str:
        """Render the anchored `ModelNew` module used by the STARK agents."""
        if backend == "cuda":
            return self._build_cuda_model_scaffold(info, level_problem)

        imports = info.imports_block.strip()
        parts = []
        if imports:
            parts.append(imports)
            parts.append("")
        parts.extend(
            [
                "# <<<IMPROVE:helpers>>>",
                "# <<<END_IMPROVE>>>",
                "",
                "class ModelNew(nn.Module):",
            ]
        )
        if info.class_preamble:
            for line in info.class_preamble.splitlines():
                parts.append(f"    {line}" if line else "")
        parts.extend(
            [
                f"    def {info.init_signature}:",
                "        super().__init__()",
                "        # <<<IMPROVE:init_body>>>",
            ]
        )
        if info.init_body:
            for line in info.init_body.splitlines():
                parts.append(f"        {line}" if line else "")
        parts.extend(
            [
                "        # <<<END_IMPROVE>>>",
                "",
                f"    def {info.forward_signature}:",
            ]
        )
        if level >= 3 and info.forward_steps:
            for index, step in enumerate(info.forward_steps, start=1):
                parts.append(f"        # <<<IMPROVE:forward_step_{index}>>>")
                if step:
                    for line in step.splitlines():
                        parts.append(f"        {line}" if line else "")
                parts.append("        # <<<END_IMPROVE>>>")
        else:
            parts.append("        # <<<IMPROVE:forward_body>>>")
            if info.forward_body:
                for line in info.forward_body.splitlines():
                    parts.append(f"        {line}" if line else "")
            parts.append("        # <<<END_IMPROVE>>>")
        parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def _build_cuda_model_scaffold(self, info: OfficialProblemInfo, level_problem: tuple[int, int]) -> str:
        imports = self._ensure_cuda_extension_imports(info.imports_block.strip())
        helper_body, cpp_body, cu_body, forward_body = self._cuda_backend_bodies(level_problem, info.forward_body)
        parts = []
        if imports:
            parts.append(imports)
            parts.append("")
        parts.extend(
            [
                "# <<<IMPROVE:helpers>>>",
            ]
        )
        for line in helper_body.splitlines():
            parts.append(line)
        parts.extend(
            [
                "# <<<END_IMPROVE>>>",
                "",
                'CUDA_CPP_SRC = r"""',
                "# <<<IMPROVE:cuda_cpp>>>",
            ]
        )
        for line in cpp_body.splitlines():
            parts.append(line)
        parts.extend(
            [
                "# <<<END_IMPROVE>>>",
                '"""',
                "",
                'CUDA_CU_SRC = r"""',
                "# <<<IMPROVE:cuda_cu>>>",
            ]
        )
        for line in cu_body.splitlines():
            parts.append(line)
        parts.extend(
            [
                "# <<<END_IMPROVE>>>",
                '"""',
                "",
                "class ModelNew(nn.Module):",
            ]
        )
        if info.class_preamble:
            for line in info.class_preamble.splitlines():
                parts.append(f"    {line}" if line else "")
        parts.extend(
            [
                f"    def {info.init_signature}:",
                "        super().__init__()",
                "        # <<<IMPROVE:init_body>>>",
            ]
        )
        if info.init_body:
            for line in info.init_body.splitlines():
                parts.append(f"        {line}" if line else "")
        parts.extend(
            [
                "        # <<<END_IMPROVE>>>",
                "",
                f"    def {info.forward_signature}:",
                "        # <<<IMPROVE:forward_body>>>",
            ]
        )
        for line in forward_body.splitlines():
            parts.append(f"        {line}" if line else "")
        parts.extend(
            [
                "        # <<<END_IMPROVE>>>",
                "",
            ]
        )
        return "\n".join(parts).rstrip() + "\n"

    @staticmethod
    def _ensure_cuda_extension_imports(imports: str) -> str:
        lines = [line for line in imports.splitlines() if line.strip()]
        required = [
            "import hashlib",
            "from torch.utils.cpp_extension import load_inline",
        ]
        existing = set(line.strip() for line in lines)
        for line in required:
            if line not in existing:
                lines.append(line)
        return "\n".join(lines)

    def _cuda_backend_bodies(self, level_problem: tuple[int, int], baseline_forward_body: str) -> tuple[str, str, str, str]:
        if level_problem == (1, 25):
            return (
                textwrap.dedent(
                    """
                    _STARK_EXTENSION = None

                    def _stark_strip_anchor_markers(source: str) -> str:
                        cleaned_lines = []
                        for line in source.splitlines():
                            stripped = line.lstrip()
                            if stripped.startswith("# <<<IMPROVE:") or stripped.startswith("# <<<END_IMPROVE>>>"):
                                continue
                            cleaned_lines.append(line)
                        return "\\n".join(cleaned_lines)

                    def _stark_extension_name() -> str:
                        digest = hashlib.sha1(
                            (_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode("utf-8")
                        ).hexdigest()[:12]
                        return f"stark_cuda_swish_{digest}"

                    def _stark_get_extension():
                        global _STARK_EXTENSION
                        if _STARK_EXTENSION is None:
                            _STARK_EXTENSION = load_inline(
                                name=_stark_extension_name(),
                                cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC),
                                cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC),
                                functions=None,
                                extra_cflags=["-O3"],
                                extra_cuda_cflags=["-O3", "--use_fast_math"],
                                with_cuda=True,
                                verbose=False,
                            )
                        return _STARK_EXTENSION
                    """
                ).strip("\n"),
                textwrap.dedent(
                    """
                    #include <torch/extension.h>

                    torch::Tensor swish_cuda(torch::Tensor x);

                    torch::Tensor swish_forward(torch::Tensor x) {
                        return swish_cuda(x);
                    }

                    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                        m.def("swish_cuda", &swish_forward, "Swish forward (CUDA)");
                    }
                    """
                ).strip("\n"),
                textwrap.dedent(
                    """
                    #include <torch/extension.h>
                    #include <cuda.h>
                    #include <cuda_runtime.h>

                    template <typename scalar_t>
                    __global__ void swish_kernel(const scalar_t* x, scalar_t* out, int64_t n) {
                        int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
                        if (index < n) {
                            scalar_t value = x[index];
                            scalar_t sigmoid = scalar_t(1) / (scalar_t(1) + exp(-value));
                            out[index] = value * sigmoid;
                        }
                    }

                    torch::Tensor swish_cuda(torch::Tensor x) {
                        TORCH_CHECK(x.is_cuda(), "swish_cuda: expected a CUDA tensor");
                        auto input = x.contiguous();
                        auto output = torch::empty_like(input);
                        int64_t n = input.numel();
                        constexpr int threads = 256;
                        const int blocks = static_cast<int>((n + threads - 1) / threads);
                        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "swish_cuda", [&] {
                            swish_kernel<scalar_t><<<blocks, threads>>>(
                                input.data_ptr<scalar_t>(),
                                output.data_ptr<scalar_t>(),
                                n
                            );
                        });
                        return output.view(input.sizes());
                    }
                    """
                ).strip("\n"),
                "return _stark_get_extension().swish_cuda(x)\n",
            )
        if level_problem == (1, 47):
            return (
                textwrap.dedent(
                    """
                    _STARK_EXTENSION = None

                    def _stark_strip_anchor_markers(source: str) -> str:
                        cleaned_lines = []
                        for line in source.splitlines():
                            stripped = line.lstrip()
                            if stripped.startswith("# <<<IMPROVE:") or stripped.startswith("# <<<END_IMPROVE>>>"):
                                continue
                            cleaned_lines.append(line)
                        return "\\n".join(cleaned_lines)

                    def _stark_extension_name() -> str:
                        digest = hashlib.sha1(
                            (_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode("utf-8")
                        ).hexdigest()[:12]
                        return f"stark_cuda_sumdim1_{digest}"

                    def _stark_get_extension():
                        global _STARK_EXTENSION
                        if _STARK_EXTENSION is None:
                            _STARK_EXTENSION = load_inline(
                                name=_stark_extension_name(),
                                cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC),
                                cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC),
                                functions=None,
                                extra_cflags=["-O3"],
                                extra_cuda_cflags=["-O3", "--use_fast_math"],
                                with_cuda=True,
                                verbose=False,
                            )
                        return _STARK_EXTENSION
                    """
                ).strip("\n"),
                textwrap.dedent(
                    """
                    #include <torch/extension.h>

                    torch::Tensor sum_dim1_keepdim_cuda(torch::Tensor x);

                    torch::Tensor sum_dim1_keepdim_forward(torch::Tensor x) {
                        return sum_dim1_keepdim_cuda(x);
                    }

                    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                        m.def("sum_dim1_keepdim_cuda", &sum_dim1_keepdim_forward, "Sum keepdim over dim=1 (CUDA)");
                    }
                    """
                ).strip("\n"),
                textwrap.dedent(
                    """
                    #include <torch/extension.h>
                    #include <cuda.h>
                    #include <cuda_runtime.h>

                    template <typename scalar_t>
                    __global__ void sum_dim1_keepdim_kernel(
                        const scalar_t* x,
                        scalar_t* out,
                        int64_t batch,
                        int64_t channels,
                        int64_t width
                    ) {
                        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
                        int64_t total = batch * width;
                        if (linear < total) {
                            int64_t batch_index = linear / width;
                            int64_t width_index = linear % width;
                            scalar_t acc = scalar_t(0);
                            int64_t base = batch_index * channels * width + width_index;
                            for (int64_t channel = 0; channel < channels; ++channel) {
                                acc += x[base + channel * width];
                            }
                            out[batch_index * width + width_index] = acc;
                        }
                    }

                    torch::Tensor sum_dim1_keepdim_cuda(torch::Tensor x) {
                        TORCH_CHECK(x.is_cuda(), "sum_dim1_keepdim_cuda: expected a CUDA tensor");
                        TORCH_CHECK(x.dim() == 3, "sum_dim1_keepdim_cuda: expected a 3D tensor");
                        auto input = x.contiguous();
                        auto output = torch::zeros({input.size(0), 1, input.size(2)}, input.options());
                        const int64_t batch = input.size(0);
                        const int64_t channels = input.size(1);
                        const int64_t width = input.size(2);
                        const int64_t total = batch * width;
                        constexpr int threads = 256;
                        const int blocks = static_cast<int>((total + threads - 1) / threads);
                        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_dim1_keepdim_cuda", [&] {
                            sum_dim1_keepdim_kernel<scalar_t><<<blocks, threads>>>(
                                input.data_ptr<scalar_t>(),
                                output.data_ptr<scalar_t>(),
                                batch,
                                channels,
                                width
                            );
                        });
                        return output;
                    }
                    """
                ).strip("\n"),
                "return _stark_get_extension().sum_dim1_keepdim_cuda(x)\n",
            )
        return self._generic_cuda_backend_bodies(level_problem, baseline_forward_body)

    @staticmethod
    def _generic_cuda_backend_bodies(level_problem: tuple[int, int], baseline_forward_body: str) -> tuple[str, str, str, str]:
        level, problem_id = level_problem
        helper_body = textwrap.dedent(
            f"""
            _STARK_EXTENSION = None

            def _stark_strip_anchor_markers(source: str) -> str:
                cleaned_lines = []
                for line in source.splitlines():
                    stripped = line.lstrip()
                    if stripped.startswith("# <<<IMPROVE:") or stripped.startswith("# <<<END_IMPROVE>>>"):
                        continue
                    cleaned_lines.append(line)
                return "\\n".join(cleaned_lines)

            def _stark_extension_name() -> str:
                digest = hashlib.sha1(
                    (_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode("utf-8")
                ).hexdigest()[:12]
                return f"stark_cuda_l{level}_p{problem_id}_{{digest}}"

            def _stark_get_extension():
                global _STARK_EXTENSION
                if _STARK_EXTENSION is None:
                    _STARK_EXTENSION = load_inline(
                        name=_stark_extension_name(),
                        cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC),
                        cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC),
                        functions=None,
                        extra_cflags=["-O3"],
                        extra_cuda_cflags=["-O3", "--use_fast_math"],
                        with_cuda=True,
                        verbose=False,
                    )
                return _STARK_EXTENSION
            """
        ).strip("\n")
        cpp_body = textwrap.dedent(
            """
            #include <torch/extension.h>

            // Replace this anchor with your pybind exports for custom CUDA entrypoints.
            // Example:
            // torch::Tensor custom_cuda(torch::Tensor x);
            // PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            //     m.def("custom_cuda", &custom_cuda, "Custom CUDA op");
            // }

            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
            """
        ).strip("\n")
        cu_body = textwrap.dedent(
            """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            // Replace this anchor with custom CUDA kernels plus their exported wrapper functions.
            // The Python forward body can call them via _stark_get_extension().your_entrypoint(...).
            """
        ).strip("\n")
        forward_lines = [
            "# Baseline fallback keeps the official PyTorch forward path.",
            "# After implementing CUDA_CPP_SRC / CUDA_CU_SRC you can switch this to _stark_get_extension().your_entrypoint(...).",
        ]
        if baseline_forward_body.strip():
            forward_lines.append(baseline_forward_body.rstrip("\n"))
        else:
            forward_lines.append("raise NotImplementedError('Empty baseline forward body for generic CUDA scaffold')")
        forward_body = "\n".join(forward_lines).rstrip() + "\n"
        return helper_body, cpp_body, cu_body, forward_body

    @staticmethod
    def _extract_grounded_regions(source_code: str) -> list[GroundedRegion]:
        pattern = re.compile(
            r"(?ms)^[ \t]*#\s*<<<IMPROVE:(?P<name>[^>]+)>>>\s*\n(?P<body>.*?)(?=^[ \t]*#\s*<<<END_IMPROVE>>>)"
        )
        regions: list[GroundedRegion] = []
        for match in pattern.finditer(source_code):
            start_line = source_code.count("\n", 0, match.start()) + 1
            end_line = source_code.count("\n", 0, match.end()) + 1
            name = match.group("name")
            excerpt = textwrap.dedent(match.group("body")).strip("\n")
            regions.append(
                GroundedRegion(
                    anchor_name=name,
                    region_role=_region_role(name),
                    start_line=start_line,
                    end_line=end_line,
                    source_excerpt=excerpt,
                    source_hash=hashlib.sha1(excerpt.encode("utf-8")).hexdigest()[:12],
                )
            )
        return regions

    def _build_cases(self, target: dict[str, Any], kind: str) -> list[TestCase]:
        torch = _require_torch()
        shapes = target[f"{kind}_shapes"]
        init_args = list(target["init_args"])
        init_kwargs = dict(target.get("init_kwargs", {}))
        cases: list[TestCase] = []
        for index, shape in enumerate(shapes, start=1):
            if target["input_kind"] == "symmetric":
                tensor = _build_symmetric_tensor(torch, shape)
            else:
                tensor = torch.rand(shape, dtype=torch.float32)
            cases.append(
                TestCase(
                    label=f"{kind}-{index}",
                    args=[tensor],
                    kwargs={},
                    init_args=list(init_args),
                    init_kwargs=dict(init_kwargs),
                )
            )
        return cases

    @staticmethod
    def _strategy_catalog_for_backend(
        level_problem: tuple[int, int],
        backend: str,
        base_catalog: list[StrategySpec],
    ) -> list[StrategySpec]:
        if backend != "cuda":
            return list(base_catalog)
        if level_problem == (1, 25):
            return [
                StrategySpec(
                    name="swish_cuda_forward_call",
                    anchor_name="forward_body",
                    strategy_summary="Keep the Swish path on the native CUDA extension while making the tensor preparation explicit.",
                    instruction="Replace the forward body with a contiguous input temporary followed by the swish_cuda extension call.",
                    expected_gain="Route the activation through the native CUDA kernel while preserving exact Swish math.",
                    good_body="x_contiguous = x.contiguous()\nreturn _stark_get_extension().swish_cuda(x_contiguous)\n",
                    broken_body="return _stark_get_extension().swish_cuda(x) + x\n",
                    debug_body="x_contiguous = x.contiguous()\nreturn _stark_get_extension().swish_cuda(x_contiguous)\n",
                    broken_failure_type="correctness_error",
                )
            ]
        if level_problem == (1, 47):
            return [
                StrategySpec(
                    name="sum_dim1_cuda_forward_call",
                    anchor_name="forward_body",
                    strategy_summary="Keep the reduction on the native CUDA extension while preserving the keepdim contract.",
                    instruction="Replace the forward body with a dim check, a contiguous input temporary, and the sum_dim1_keepdim_cuda extension call.",
                    expected_gain="Route the reduction through the native CUDA kernel without changing the output shape contract.",
                    good_body=(
                        "if self.dim != 1:\n"
                        "    return torch.sum(x, dim=self.dim, keepdim=True)\n"
                        "x_contiguous = x.contiguous()\n"
                        "return _stark_get_extension().sum_dim1_keepdim_cuda(x_contiguous)\n"
                    ),
                    broken_body="return _stark_get_extension().sum_dim1_keepdim_cuda(x).squeeze(1)\n",
                    debug_body=(
                        "if self.dim != 1:\n"
                        "    return torch.sum(x, dim=self.dim, keepdim=True)\n"
                        "x_contiguous = x.contiguous()\n"
                        "return _stark_get_extension().sum_dim1_keepdim_cuda(x_contiguous)\n"
                    ),
                    broken_failure_type="correctness_error",
                )
            ]
        return list(base_catalog)

    @staticmethod
    def _load_python_source(path: Path, function_name: str, label: str) -> str:
        if not path.exists():
            raise BridgeLoadError(f"{label.capitalize()} source file does not exist: {path}")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BridgeLoadError(f"Failed to read {label} source file '{path}': {exc}") from exc
        try:
            module = ast.parse(content, filename=str(path))
        except SyntaxError as exc:
            location = f"{path}:{exc.lineno}:{exc.offset}"
            raise BridgeLoadError(f"{label.capitalize()} source has invalid Python syntax at {location}: {exc.msg}") from exc
        if not KernelBenchTaskBridge._has_function(module, function_name):
            raise BridgeLoadError(f"{label.capitalize()} function '{function_name}' was not found in {path}")
        return content

    @staticmethod
    def _has_function(module: ast.AST, function_name: str) -> bool:
        for node in getattr(module, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return True
        return False


def _is_super_init_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    if isinstance(call.func, ast.Attribute) and call.func.attr == "__init__":
        inner = call.func.value
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "super":
            return True
    return False


def _build_symmetric_tensor(torch_module, shape: tuple[int, ...]):
    total = 1
    for dim in shape:
        total *= dim
    values = torch_module.linspace(-4.0, 4.0, total, dtype=torch_module.float32)
    return values.reshape(shape)


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise BridgeLoadError(f"torch is required for KernelBench task presets: {exc}") from exc
    return torch
