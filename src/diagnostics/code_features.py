from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_TRANSCENDENTAL_PATTERN = re.compile(r"\b(exp|sigmoid|tanh|gelu|silu|sin|cos|log|sqrt|rsqrt)\b")


@dataclass(frozen=True)
class StaticCodeFeaturesResult:
    supported: bool
    features: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def derive_static_code_features(task: Any) -> StaticCodeFeaturesResult:
    semantic = getattr(task, "semantic_profile", None)
    op_type = str(getattr(semantic, "op_type", "unknown") or "unknown")
    reference_code = str(getattr(task, "reference_code", "") or "")
    exact_facts = getattr(semantic, "exact_facts", None)
    exact_kind = str(getattr(exact_facts, "kind", "unknown") or "unknown")

    base = {
        "has_vector_load_store": False,
        "has_shared_memory_tile": False,
        "uses_vector_types": False,
        "has_bounds_check": True,
        "tc_eligible": False,
        "is_aligned_vector_access": True,
        "has_tail_handling_overhead": False,
        "has_multiple_kernels_in_forward": False,
        "cudagraph_eligible": False,
        "uses_transcendentals": bool(_TRANSCENDENTAL_PATTERN.search(reference_code.lower())),
    }

    if op_type in {"elementwise", "loss"}:
        return StaticCodeFeaturesResult(
            supported=True,
            features={
                **base,
                "has_reuse": False,
                "streaming_no_reuse": True,
                "is_pointwise": True,
                "is_naive_gemm": False,
                "has_k_loop": False,
                "is_gemm_kloop": False,
                "is_stencil_conv": False,
                "kernel_structure_id": 0,
            },
            notes=["Mapped to S0 streaming structure from semantic op_type."],
        )
    if op_type == "matmul":
        return StaticCodeFeaturesResult(
            supported=True,
            features={
                **base,
                "has_reuse": True,
                "streaming_no_reuse": False,
                "is_pointwise": False,
                "is_naive_gemm": True,
                "has_k_loop": True,
                "is_gemm_kloop": True,
                "is_stencil_conv": False,
                "kernel_structure_id": 1,
            },
            notes=["Mapped to S1 reuse-friendly GEMM structure from semantic op_type."],
        )
    if op_type == "convolution":
        return StaticCodeFeaturesResult(
            supported=True,
            features={
                **base,
                "has_reuse": True,
                "streaming_no_reuse": False,
                "is_pointwise": False,
                "is_naive_gemm": False,
                "has_k_loop": True,
                "is_gemm_kloop": False,
                "is_stencil_conv": True,
                "kernel_structure_id": 1,
            },
            notes=["Mapped to S1 reuse-friendly stencil/conv structure from semantic op_type."],
        )
    if op_type == "reduction":
        return StaticCodeFeaturesResult(
            supported=True,
            features={
                **base,
                "has_reuse": False,
                "streaming_no_reuse": False,
                "is_pointwise": False,
                "is_naive_gemm": False,
                "has_k_loop": False,
                "is_gemm_kloop": False,
                "is_stencil_conv": False,
                "kernel_structure_id": 3,
            },
            notes=["Mapped to S3 reduction/scan structure from semantic op_type."],
        )
    if op_type == "normalization":
        return StaticCodeFeaturesResult(
            supported=True,
            features={
                **base,
                "has_reuse": True,
                "streaming_no_reuse": False,
                "is_pointwise": False,
                "is_naive_gemm": False,
                "has_k_loop": False,
                "is_gemm_kloop": False,
                "is_stencil_conv": False,
                "kernel_structure_id": 3,
            },
            notes=[
                "Mapped to S3 reduction/scan structure from semantic op_type.",
                f"Exact semantic fact: {exact_kind}.",
            ]
            if exact_kind != "unknown"
            else ["Mapped to S3 reduction/scan structure from semantic op_type."],
        )
    return StaticCodeFeaturesResult(
        supported=False,
        notes=[f"Static machine-check code features are not enabled for op_type={op_type}."],
    )
