from __future__ import annotations

"""Worker process for isolated KernelBench candidate evaluation."""

import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ..models import StarkConfig, TaskSpec
from .evaluator_paper import KernelBenchPaperEvaluator


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m src.evaluation.worker INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    input_path = Path(argv[0])
    output_path = Path(argv[1])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    task = _task_from_payload(payload["task"])
    config = _config_from_payload(payload["config"])
    code = str(payload["code"])
    result = KernelBenchPaperEvaluator().evaluate(task, code, config)
    output_path.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
    return 0


def _task_from_payload(payload: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        name=str(payload.get("name", "kernelbench_task")),
        description=str(payload.get("description", "")),
        source_code=str(payload.get("source_code", "")),
        reference_code=str(payload.get("reference_code", "")),
        function_name=str(payload.get("function_name", "ModelNew")),
        reference_function_name=str(payload.get("reference_function_name", "Model")),
        test_cases=[],
        benchmark_cases=[],
        tags=list(payload.get("tags") or []),
        source_origin=payload.get("source_origin"),
        benchmark_family=payload.get("benchmark_family"),
        entry_kind=str(payload.get("entry_kind", "model_class")),
        level=payload.get("level"),
        problem_id=payload.get("problem_id"),
        backend=payload.get("backend"),
        source_root=payload.get("source_root"),
    )


def _config_from_payload(payload: dict[str, Any]) -> StarkConfig:
    allowed = {field.name for field in fields(StarkConfig)}
    return StarkConfig(**{key: value for key, value in dict(payload).items() if key in allowed})


if __name__ == "__main__":
    raise SystemExit(main())
