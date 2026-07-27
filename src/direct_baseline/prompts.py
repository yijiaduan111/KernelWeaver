"""Prompt construction for the direct LLM baseline."""

from __future__ import annotations

from ..models import TaskSpec


def build_direct_system_prompt(backend: str) -> str:
    return (
        "You are an expert GPU kernel engineer solving one KernelBench task directly.\n"
        "Return exactly one complete Python source file and nothing else.\n"
        "The file must define class ModelNew(nn.Module) with the same public behavior as the official Model.\n"
        "Do not return markdown commentary, explanations, plans, JSON, tool calls, shell commands, or partial patches.\n"
        "Do not depend on external files or prior conversation state.\n"
        f"Target backend: {backend}."
    )


def build_direct_user_payload(task: TaskSpec, backend: str) -> dict:
    return {
        "task_name": task.name,
        "benchmark_family": task.benchmark_family,
        "level": task.level,
        "problem_id": task.problem_id,
        "backend": backend,
        "entry_contract": {
            "required_class": "ModelNew",
            "reference_class": "Model",
            "entry_kind": task.entry_kind,
            "function_name": task.function_name,
            "reference_function_name": task.reference_function_name,
        },
        "constraints": [
            "Return a complete Python file that can be passed directly to the official KernelBench evaluator as custom_model_src.",
            "The file must include all imports and helper definitions it needs.",
            "Keep ModelNew.__init__ and ModelNew.forward compatible with the official Model constructor and forward method.",
            "If the reference uses super(Model, self), update it so the returned ModelNew source is self-contained and importable.",
            "For CUDA, torch.utils.cpp_extension.load_inline is allowed; keep kernels self-contained inside the returned Python source.",
            "Correctness is mandatory; optimize only when the implementation remains numerically compatible with the reference.",
            "Do not use STARK anchors, region patches, semantic profiles, deliberation strategies, diagnostics, or feedback history.",
        ],
        "official_reference_source": task.reference_code,
    }
