"""Raw execution facts extracted from KernelBench factories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TensorFact:
    name: str
    shape: list[int] = field(default_factory=list)
    dtype: str = ""
    numel: int = 0


@dataclass
class ExecutionFacts:
    input_tensors: list[TensorFact] = field(default_factory=list)
    init_tensors: list[TensorFact] = field(default_factory=list)
    init_args: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ExecutionFacts | None":
        if not payload:
            return None
        input_tensors = [_tensor_fact_from_dict(item) for item in payload.get("input_tensors", []) if isinstance(item, dict)]
        init_tensors = [_tensor_fact_from_dict(item) for item in payload.get("init_tensors", []) if isinstance(item, dict)]
        raw_init_args = _normalize_json_like(payload.get("init_args", []))
        init_args = raw_init_args if isinstance(raw_init_args, list) else [raw_init_args]
        return cls(
            input_tensors=input_tensors,
            init_tensors=init_tensors,
            init_args=init_args,
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "input_tensors": [_tensor_fact_to_prompt(item) for item in self.input_tensors[:4]],
            "init_tensors": [_tensor_fact_to_prompt(item) for item in self.init_tensors[:4]],
            "init_args": _normalize_json_like(self.init_args[:8]),
        }


def _tensor_fact_from_dict(payload: dict[str, Any]) -> TensorFact:
    return TensorFact(
        name=str(payload.get("name", "")),
        shape=[int(dim) for dim in payload.get("shape", [])],
        dtype=str(payload.get("dtype", "")),
        numel=int(payload.get("numel", 0)),
    )


def _tensor_fact_to_prompt(item: TensorFact) -> dict[str, Any]:
    return {
        "name": item.name,
        "shape": list(item.shape),
        "dtype": item.dtype,
        "numel": item.numel,
    }


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
