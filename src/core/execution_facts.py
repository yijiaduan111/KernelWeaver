"""Deterministic execution-shape facts extracted from KernelBench factories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any


@dataclass
class TensorFact:
    name: str
    shape: list[int] = field(default_factory=list)
    dtype: str = ""
    device: str = ""
    numel: int = 0


@dataclass
class ExecutionDerived:
    primary_shape: list[int] = field(default_factory=list)
    tensor_count: int = 0
    has_broadcast_inputs: bool = False
    outer_size: int | None = None
    inner_size: int | None = None


@dataclass
class ExecutionFacts:
    input_tensors: list[TensorFact] = field(default_factory=list)
    init_tensors: list[TensorFact] = field(default_factory=list)
    init_args: list[Any] = field(default_factory=list)
    derived: ExecutionDerived = field(default_factory=ExecutionDerived)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ExecutionFacts | None:
        if not payload:
            return None
        input_tensors = [_tensor_fact_from_dict(item) for item in payload.get("input_tensors", []) if isinstance(item, dict)]
        init_tensors = [_tensor_fact_from_dict(item) for item in payload.get("init_tensors", []) if isinstance(item, dict)]
        raw_init_args = _normalize_json_like(payload.get("init_args", []))
        init_args = raw_init_args if isinstance(raw_init_args, list) else [raw_init_args]
        derived_payload = payload.get("derived") if isinstance(payload.get("derived"), dict) else {}
        derived = ExecutionDerived(
            primary_shape=[int(dim) for dim in derived_payload.get("primary_shape", [])],
            tensor_count=int(derived_payload.get("tensor_count", len(input_tensors) + len(init_tensors))),
            has_broadcast_inputs=bool(derived_payload.get("has_broadcast_inputs", False)),
            outer_size=_maybe_int(derived_payload.get("outer_size")),
            inner_size=_maybe_int(derived_payload.get("inner_size")),
        )
        return cls(
            input_tensors=input_tensors,
            init_tensors=init_tensors,
            init_args=init_args,
            derived=derived,
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "input_tensors": [_tensor_fact_to_prompt(item) for item in self.input_tensors[:4]],
            "init_tensors": [_tensor_fact_to_prompt(item) for item in self.init_tensors[:4]],
            "init_args": _normalize_json_like(self.init_args[:8]),
        }


def build_execution_facts(input_tensors: list[TensorFact], init_tensors: list[TensorFact]) -> ExecutionFacts:
    all_tensors = [*input_tensors, *init_tensors]
    primary = input_tensors[0] if input_tensors else (all_tensors[0] if all_tensors else None)
    primary_shape = list(primary.shape) if primary else []
    outer_size, inner_size = _shape_outer_inner(primary_shape)
    has_broadcast_inputs = _has_broadcast_inputs(input_tensors)
    return ExecutionFacts(
        input_tensors=input_tensors,
        init_tensors=init_tensors,
        derived=ExecutionDerived(
            primary_shape=primary_shape,
            tensor_count=len(all_tensors),
            has_broadcast_inputs=has_broadcast_inputs,
            outer_size=outer_size,
            inner_size=inner_size,
        ),
    )


def infer_workload_profile(op_type: str, execution_facts: ExecutionFacts | None) -> tuple[str | None, str | None]:
    if execution_facts is None:
        return None, None
    derived = execution_facts.derived
    has_tensor_facts = bool(execution_facts.input_tensors or execution_facts.init_tensors or derived.primary_shape)
    outer_size = derived.outer_size
    inner_size = derived.inner_size
    if not has_tensor_facts:
        return "unknown", "unknown"
    if op_type in {"normalization", "reduction"}:
        if outer_size is None or inner_size is None or outer_size <= 0 or inner_size <= 0:
            return "unknown", "unknown"
        if outer_size < 64 and inner_size > 512000:
            return "few_rows_huge_inner", "memory_bound"
        if outer_size < 64 and inner_size > 512:
            return "few_rows_medium_inner", "unknown"
        if outer_size < 64:
            return "few_rows_small_inner", "launch_overhead"
        return "many_rows_small_inner", "unknown"
    if op_type == "matmul":
        return "regular_matmul", "compute_bound"
    if op_type == "convolution":
        return "spatial_conv", "unknown"
    if op_type == "pooling":
        return "spatial_pooling", "unknown"
    if op_type == "elementwise":
        tag = "elementwise_broadcast" if derived.has_broadcast_inputs else "elementwise_dense"
        return tag, "unknown"
    return "unknown", "unknown"


def _tensor_fact_from_dict(payload: dict[str, Any]) -> TensorFact:
    return TensorFact(
        name=str(payload.get("name", "")),
        shape=[int(dim) for dim in payload.get("shape", [])],
        dtype=str(payload.get("dtype", "")),
        device=str(payload.get("device", "")),
        numel=int(payload.get("numel", 0)),
    )


def _tensor_fact_to_prompt(item: TensorFact) -> dict[str, Any]:
    return {
        "name": item.name,
        "shape": list(item.shape),
        "dtype": item.dtype,
        "device": item.device,
        "numel": item.numel,
    }


def _shape_outer_inner(shape: list[int]) -> tuple[int | None, int | None]:
    if not shape:
        return None, None
    if len(shape) == 1:
        return 1, int(shape[0])
    return int(prod(shape[:-1])), int(shape[-1])


def _has_broadcast_inputs(tensors: list[TensorFact]) -> bool:
    shapes = [tuple(item.shape) for item in tensors if item.shape]
    if len(shapes) < 2:
        return False
    first = shapes[0]
    return any(shape != first for shape in shapes[1:])


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_json_like(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_like(item) for key, item in value.items()}
    return repr(value)
