"""Exact operator-specific interpreters over raw execution facts."""

from __future__ import annotations

import ast
import textwrap
from typing import Any

from ..core.execution_facts import ExecutionFacts
from .schema import SemanticFactProfile


_EXACT_REDUCTION_CALLS = {
    "torch.sum",
    "sum",
    "torch.mean",
    "mean",
    "torch.amax",
    "amax",
    "torch.amin",
    "amin",
}


def derive_exact_semantic_facts(problem_info: Any, op_type: str, execution_facts: ExecutionFacts | None) -> SemanticFactProfile | None:
    if execution_facts is None:
        return None
    if op_type == "normalization":
        return _derive_normalization_facts(problem_info, execution_facts)
    if op_type == "reduction":
        return _derive_reduction_facts(problem_info, execution_facts)
    return None


def _derive_normalization_facts(problem_info: Any, execution_facts: ExecutionFacts) -> SemanticFactProfile | None:
    input_shapes = [list(item.shape) for item in execution_facts.input_tensors if item.shape]
    init_args = execution_facts.init_args
    if len(input_shapes) != 1:
        return None
    input_shape = input_shapes[0]
    if not input_shape:
        return None

    normalized_shape = _extract_layernorm_normalized_shape(problem_info, init_args)
    if normalized_shape is not None:
        if len(normalized_shape) > len(input_shape):
            return None
        tail = input_shape[-len(normalized_shape):] if normalized_shape else []
        if list(tail) != list(normalized_shape):
            return None
        return SemanticFactProfile(
            kind="layernorm_exact_axes",
            confidence="exact",
            details={
                "input_shape": input_shape,
                "normalized_shape": list(normalized_shape),
                "outer_extent": _prod(input_shape[:-len(normalized_shape)]) if normalized_shape else _prod(input_shape),
                "inner_extent": _prod(normalized_shape) if normalized_shape else 1,
                "normalized_rank": len(normalized_shape),
            },
            notes=[
                "Exact LayerNorm axes recovered from get_init_inputs() and input tensor shape.",
                "Only emitted when normalized_shape exactly matches the input tail.",
            ],
        )

    batchnorm_channels = _extract_batchnorm_num_features(problem_info, init_args)
    if batchnorm_channels is not None:
        if len(input_shape) < 2 or input_shape[1] != batchnorm_channels:
            return None
        return SemanticFactProfile(
            kind="batchnorm_channel_axes",
            confidence="exact",
            details={
                "input_shape": input_shape,
                "channel_dim": 1,
                "channel_extent": batchnorm_channels,
                "spatial_shape": input_shape[2:],
            },
            notes=[
                "Exact BatchNorm channel axis recovered from get_init_inputs() and input tensor shape.",
                "Only emitted for canonical channel-first BatchNorm inputs.",
            ],
        )

    return None


def _derive_reduction_facts(problem_info: Any, execution_facts: ExecutionFacts) -> SemanticFactProfile | None:
    input_shapes = [list(item.shape) for item in execution_facts.input_tensors if item.shape]
    init_args = execution_facts.init_args
    if len(input_shapes) != 1:
        return None
    input_shape = input_shapes[0]
    if not input_shape:
        return None
    call = _extract_reduction_call(str(getattr(problem_info, "forward_body", "") or ""))
    if call is None:
        return None
    reduce_dim = _resolve_exact_reduction_dim(call["dim"], init_args)
    if reduce_dim is None:
        return None
    rank = len(input_shape)
    canonical_dim = reduce_dim if reduce_dim >= 0 else reduce_dim + rank
    if canonical_dim < 0 or canonical_dim >= rank:
        return None
    return SemanticFactProfile(
        kind="reduction_exact_axis",
        confidence="exact",
        details={
            "input_shape": input_shape,
            "reduce_dim": canonical_dim,
            "keepdim": call["keepdim"],
            "reduced_extent": int(input_shape[canonical_dim]),
        },
        notes=[
            "Exact reduction axis recovered from forward() and get_init_inputs().",
            "Only emitted when dim and keepdim are statically provable.",
        ],
    )


def _extract_layernorm_normalized_shape(problem_info: Any, init_args: list[Any]) -> list[int] | None:
    if len(init_args) != 1:
        return None
    normalized = init_args[0]
    if not isinstance(normalized, list) or not normalized:
        return None
    if not all(isinstance(dim, int) for dim in normalized):
        return None
    text = _problem_text(problem_info)
    if "layernorm" not in text and "layer_norm" not in text:
        return None
    return [int(dim) for dim in normalized]


def _extract_batchnorm_num_features(problem_info: Any, init_args: list[Any]) -> int | None:
    if len(init_args) != 1 or not isinstance(init_args[0], int):
        return None
    text = _problem_text(problem_info)
    if "batchnorm" not in text and "batch_norm" not in text:
        return None
    return int(init_args[0])


def _problem_text(problem_info: Any) -> str:
    init_text = str(getattr(problem_info, "init_body", "") or "")
    forward_text = str(getattr(problem_info, "forward_body", "") or "")
    path_text = str(getattr(problem_info, "path", "") or "")
    desc_text = str(getattr(problem_info, "description", "") or "")
    return "\n".join([init_text, forward_text, path_text, desc_text]).lower()


def _resolve_exact_reduction_dim(dim_value: Any, init_args: list[Any]) -> int | None:
    if isinstance(dim_value, int):
        return dim_value
    if dim_value == "self.dim" and len(init_args) == 1 and isinstance(init_args[0], int):
        return int(init_args[0])
    return None


def _extract_reduction_call(forward_body: str) -> dict[str, Any] | None:
    if not forward_body.strip():
        return None
    try:
        module = ast.parse(textwrap.dedent(forward_body))
    except SyntaxError:
        return None
    for node in module.body:
        value = node.value if isinstance(node, ast.Return) else None
        if not isinstance(value, ast.Call):
            continue
        call_name = _call_name(value.func)
        if call_name not in _EXACT_REDUCTION_CALLS:
            continue
        dim = None
        keepdim = False
        if len(value.args) >= 2 and isinstance(value.args[1], ast.Constant) and isinstance(value.args[1].value, int):
            dim = int(value.args[1].value)
        for kw in value.keywords:
            if kw.arg == "dim":
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                    dim = int(kw.value.value)
                elif isinstance(kw.value, ast.Attribute) and isinstance(kw.value.value, ast.Name) and kw.value.value.id == "self" and kw.value.attr == "dim":
                    dim = "self.dim"
            elif kw.arg == "keepdim":
                if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
                    return None
                keepdim = bool(kw.value.value)
        if dim is None:
            return None
        return {"dim": dim, "keepdim": keepdim}
    return None


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return None


def _prod(values: list[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return int(total)
