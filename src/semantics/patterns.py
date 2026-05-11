"""Rule patterns for lightweight KernelBench semantic analysis."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class PatternMatch:
    op_type: str
    op_names: list[str] = field(default_factory=list)


_ELEMENTWISE_TOKENS = {
    "relu", "leaky_relu", "sigmoid", "tanh", "gelu", "silu", "swish", "hardswish", "hard_swish",
    "exp", "log", "sqrt", "abs", "sin", "cos", "mul", "add", "sub", "div",
}
_REDUCTION_TOKENS = {"sum", "mean", "amax", "amin", "max", "min", "prod", "argmax", "argmin"}
_NORM_TOKENS = {"layer_norm", "layernorm", "batch_norm", "batchnorm", "group_norm", "instancenorm", "rms_norm"}
_MATMUL_TOKENS = {"matmul", "mm", "bmm", "linear", "einsum", "addmm"}
_CONV_TOKENS = {"conv", "conv1d", "conv2d", "conv3d", "conv_transpose", "depthwise"}
_POOL_TOKENS = {"pool", "max_pool", "avg_pool", "adaptive_avg_pool", "adaptive_max_pool"}
_ATTENTION_TOKENS = {"attention", "scaled_dot_product_attention", "qkv"}
_LOSS_TOKENS = {"loss", "cross_entropy", "nll_loss", "mse_loss", "bce"}


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []
        self.has_matmul_operator = False
        self.has_arithmetic_operator = False

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name:
            self.names.append(name)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self.has_matmul_operator = True
        else:
            self.has_arithmetic_operator = True
        self.generic_visit(node)


def classify_statement(source: str, task_hint: str = "") -> PatternMatch:
    text = f"{task_hint}\n{source}".lower()
    names = _collect_call_names(source)
    tokens = _normalized_tokens(text, names)
    op_type = _classify_tokens(tokens, text)
    return PatternMatch(op_type=op_type, op_names=sorted(tokens))


def _collect_call_names(source: str) -> set[str]:
    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        collector = _CallCollector()
        collector.visit(tree)
        for name in collector.names:
            names.add(name.lower())
            names.add(name.split(".")[-1].lower())
        if collector.has_matmul_operator:
            names.add("matmul")
        if collector.has_arithmetic_operator:
            names.add("arithmetic")
    return names


def _normalized_tokens(text: str, names: set[str]) -> set[str]:
    tokens = set(names)
    for raw in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text):
        tokens.add(raw.lower())
    if "swish" in text or ("sigmoid" in tokens and ("arithmetic" in tokens or "*" in text)):
        tokens.add("swish")
    if "layernorm" in text:
        tokens.add("layer_norm")
    if "batchnorm" in text:
        tokens.add("batch_norm")
    if "crossentropyloss" in text:
        tokens.add("cross_entropy")
    return tokens


def _classify_tokens(tokens: set[str], text: str) -> str:
    if tokens & _LOSS_TOKENS:
        return "loss"
    if tokens & _ATTENTION_TOKENS:
        return "attention"
    if tokens & _NORM_TOKENS or ("mean" in tokens and ("var" in tokens or "std" in tokens)):
        return "normalization"
    if tokens & _CONV_TOKENS:
        return "convolution"
    if tokens & _POOL_TOKENS:
        return "pooling"
    if tokens & _MATMUL_TOKENS:
        return "matmul"
    if tokens & _REDUCTION_TOKENS:
        return "reduction"
    if tokens & _ELEMENTWISE_TOKENS or "arithmetic" in tokens:
        return "elementwise"
    return "unknown"


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return None
