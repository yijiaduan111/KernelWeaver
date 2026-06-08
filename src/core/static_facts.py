"""Static KernelBench facts extraction without executing factory functions."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .execution_facts import ExecutionFacts, TensorFact


class _Unknown:
    pass


_UNKNOWN = _Unknown()


class StaticValueEvaluator:
    def __init__(self, module: ast.Module) -> None:
        self.env: dict[str, Any] = {}
        for node in module.body:
            if isinstance(node, ast.Assign):
                value = self._safe_eval(node.value, self.env)
                if value is _UNKNOWN:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.env[target.id] = value

    def evaluate_function_return(self, func: ast.FunctionDef | None) -> Any:
        if func is None:
            return None
        local_env = dict(self.env)
        for node in func.body:
            if isinstance(node, ast.Assign):
                value = self._safe_eval(node.value, local_env)
                if value is _UNKNOWN:
                    return None
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        local_env[target.id] = value
                continue
            if isinstance(node, ast.Return):
                value = self._safe_eval(node.value, local_env)
                return None if value is _UNKNOWN else value
            return None
        return None

    def _safe_eval(self, node: ast.AST | None, env: dict[str, Any]) -> Any:
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id, _UNKNOWN)
        if isinstance(node, ast.Tuple):
            items = [self._safe_eval(item, env) for item in node.elts]
            return _UNKNOWN if any(item is _UNKNOWN for item in items) else tuple(items)
        if isinstance(node, ast.List):
            items = [self._safe_eval(item, env) for item in node.elts]
            return _UNKNOWN if any(item is _UNKNOWN for item in items) else items
        if isinstance(node, ast.Dict):
            keys = [self._safe_eval(item, env) for item in node.keys]
            values = [self._safe_eval(item, env) for item in node.values]
            if any(item is _UNKNOWN for item in [*keys, *values]):
                return _UNKNOWN
            return {str(key): value for key, value in zip(keys, values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._safe_eval(node.operand, env)
            return _UNKNOWN if value is _UNKNOWN else -value
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node, env)
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        if isinstance(node, ast.Starred):
            value = self._safe_eval(node.value, env)
            if isinstance(value, (list, tuple)):
                return list(value)
            return _UNKNOWN
        return _UNKNOWN

    def _eval_binop(self, node: ast.BinOp, env: dict[str, Any]) -> Any:
        left = self._safe_eval(node.left, env)
        right = self._safe_eval(node.right, env)
        if left is _UNKNOWN or right is _UNKNOWN:
            return _UNKNOWN
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return _UNKNOWN
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        return _UNKNOWN

    def _eval_call(self, node: ast.Call, env: dict[str, Any]) -> Any:
        call_name = _call_name(node.func)
        if call_name not in {"torch.rand", "torch.randn", "torch.randint", "rand", "randn", "randint"}:
            return _UNKNOWN
        args = _resolve_call_args(node.args, lambda item: self._safe_eval(item, env))
        if args is _UNKNOWN:
            return _UNKNOWN
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                return _UNKNOWN
            value = self._safe_eval(kw.value, env)
            if value is _UNKNOWN:
                return _UNKNOWN
            kwargs[kw.arg] = value
        if call_name in {"torch.randint", "randint"}:
            if len(args) < 3:
                return _UNKNOWN
            shape = _coerce_shape(args[2])
            if shape is None:
                return _UNKNOWN
            dtype = "int64"
        else:
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                shape = _coerce_shape(args[0])
            else:
                shape = _coerce_shape(args)
            if shape is None:
                return _UNKNOWN
            dtype = _dtype_name(kwargs.get("dtype"), default="float32")
        return {
            "kind": "tensor",
            "shape": shape,
            "dtype": dtype,
            "numel": _numel(shape),
        }


def extract_execution_facts_from_path(path: Path) -> ExecutionFacts | None:
    source = path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None
    evaluator = StaticValueEvaluator(module)
    inputs_func = _find_top_level_function(module, "get_inputs")
    init_func = _find_top_level_function(module, "get_init_inputs")
    input_values = evaluator.evaluate_function_return(inputs_func) if inputs_func is not None else None
    init_values = evaluator.evaluate_function_return(init_func) if init_func is not None else None
    if input_values is None and init_values is None:
        return None
    input_tensors = extract_tensor_facts_from_value("input", input_values)
    init_tensors = extract_tensor_facts_from_value("init", init_values)
    init_args = _normalize_json_like(init_values if isinstance(init_values, list) else [init_values] if init_values is not None else [])
    normalized_init_args = init_args if isinstance(init_args, list) else [init_args]
    if not input_tensors and not init_tensors and len(normalized_init_args) == 0:
        return None
    return ExecutionFacts(
        input_tensors=input_tensors,
        init_tensors=init_tensors,
        init_args=normalized_init_args,
    )


def extract_tensor_facts_from_value(label: str, value: Any) -> list[TensorFact]:
    output: list[TensorFact] = []

    def walk(prefix: str, item: Any) -> None:
        if isinstance(item, dict) and item.get("kind") == "tensor":
            output.append(
                TensorFact(
                    name=prefix,
                    shape=[int(dim) for dim in item.get("shape", [])],
                    dtype=str(item.get("dtype", "")),
                    numel=int(item.get("numel", 0)),
                )
            )
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                walk(f"{prefix}_{index}", child)
            return
        if isinstance(item, tuple):
            for index, child in enumerate(item):
                walk(f"{prefix}_{index}", child)
            return
        if isinstance(item, dict):
            for key in sorted(item):
                walk(f"{prefix}_{key}", item[key])

    if value is not None:
        walk(label, value)
    return output


def _find_top_level_function(module: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _resolve_call_args(nodes: list[ast.AST], resolve) -> Any:
    args: list[Any] = []
    for node in nodes:
        value = resolve(node)
        if value is _UNKNOWN:
            return _UNKNOWN
        if isinstance(node, ast.Starred):
            if not isinstance(value, list):
                return _UNKNOWN
            args.extend(value)
        else:
            args.append(value)
    return args


def _coerce_shape(value: Any) -> list[int] | None:
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    shape: list[int] = []
    for dim in value:
        if not isinstance(dim, int):
            return None
        shape.append(int(dim))
    return shape


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _dtype_name(value: Any, default: str = "float32") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.replace("torch.", "")
    return default


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return None


def _normalize_json_like(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_like(item) for key, item in value.items() if key != "kind"}
    return repr(value)
