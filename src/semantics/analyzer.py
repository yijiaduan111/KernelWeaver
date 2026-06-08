"""Deterministic semantic analysis for KernelBench tasks."""

from __future__ import annotations

from typing import Any

from ..core.execution_facts import ExecutionFacts
from .exact_facts import derive_exact_semantic_facts
from .patterns import classify_statement
from .schema import OptimizationIntent, SemanticAnchorProfile, SemanticFactProfile, SemanticProfile


class SemanticAnalyzer:
    """Attach optimization semantics to loader-generated anchors."""

    def analyze(
        self,
        problem_info: Any,
        grounded_regions: list[Any],
        backend: str,
        execution_facts: ExecutionFacts | None = None,
        mode: str = "rule",
        max_anchor_hints: int = 6,
    ) -> SemanticProfile:
        if mode != "rule":
            return SemanticProfile(enabled=True, mode=mode, op_type="unknown", summary=f"Unsupported semantic mode: {mode}")
        task_hint = " ".join(
            str(item or "")
            for item in [
                getattr(problem_info, "path", ""),
                getattr(problem_info, "description", ""),
                getattr(problem_info, "init_body", ""),
                getattr(problem_info, "forward_body", ""),
            ]
        )
        anchors = self._anchor_profiles(problem_info, grounded_regions, backend, task_hint)
        forward_anchors = [anchor for anchor in anchors if anchor.region_role == "forward"]
        task_op_type = self._task_op_type(forward_anchors, task_hint)
        recommended = self._recommended_anchors(anchors, backend, max_anchor_hints)
        exact_facts = derive_exact_semantic_facts(problem_info, task_op_type, execution_facts)
        intents = _intents_for(task_op_type, recommended, backend)
        risks = _risk_notes_for(task_op_type, exact_facts)
        return SemanticProfile(
            enabled=True,
            mode="rule",
            op_type=task_op_type,
            summary=_summary_for(task_op_type, exact_facts),
            source=str(getattr(problem_info, "path", "")) or None,
            exact_facts=exact_facts,
            recommended_anchors=recommended,
            anchors=anchors[:max_anchor_hints],
            optimization_intents=intents,
            risk_notes=risks,
        )

    def _anchor_profiles(self, problem_info: Any, grounded_regions: list[Any], backend: str, task_hint: str) -> list[SemanticAnchorProfile]:
        forward_steps = list(getattr(problem_info, "forward_steps", []) or [])
        profiles: list[SemanticAnchorProfile] = []
        for region in grounded_regions:
            name = str(getattr(region, "anchor_name", ""))
            role = str(getattr(region, "region_role", "unknown"))
            source = str(getattr(region, "source_excerpt", ""))
            if name.startswith("forward_stmt_"):
                source = _forward_step_source(name, forward_steps) or source
            match = classify_statement(source, task_hint=task_hint if role == "forward" else "")
            semantic_type = match.op_type if role == "forward" else _non_forward_type(name)
            profiles.append(
                SemanticAnchorProfile(
                    anchor_name=name,
                    region_role=role,
                    semantic_type=semantic_type,
                    source_excerpt=source,
                    op_names=match.op_names[:12],
                    optimization_intents=_anchor_intent_names(semantic_type),
                    backend_hints=_anchor_backend_hints(semantic_type, backend),
                    risk_notes=_risk_notes_for(semantic_type, None),
                    priority=_anchor_priority(name, role, semantic_type),
                )
            )
        return sorted(profiles, key=lambda item: (-item.priority, item.anchor_name))

    @staticmethod
    def _task_op_type(forward_anchors: list[SemanticAnchorProfile], task_hint: str) -> str:
        if not forward_anchors:
            return classify_statement(task_hint).op_type
        priority = {
            "attention": 90,
            "loss": 80,
            "normalization": 70,
            "convolution": 65,
            "pooling": 60,
            "matmul": 55,
            "reduction": 50,
            "elementwise": 40,
            "unknown": 0,
        }
        return max(forward_anchors, key=lambda item: priority.get(item.semantic_type, 0)).semantic_type

    @staticmethod
    def _recommended_anchors(anchors: list[SemanticAnchorProfile], backend: str, max_anchor_hints: int) -> list[str]:
        names: list[str] = []
        for anchor in anchors:
            if anchor.region_role == "forward" and anchor.semantic_type != "unknown":
                names.append(anchor.anchor_name)
        if backend == "cuda":
            for required in ["cuda_cu", "cuda_cpp"]:
                if any(anchor.anchor_name == required for anchor in anchors):
                    names.append(required)
        else:
            for anchor in anchors:
                if anchor.anchor_name.endswith("_kernel"):
                    names.append(anchor.anchor_name)
        for anchor in anchors:
            if anchor.anchor_name not in names and anchor.region_role in {"forward", "helper"}:
                names.append(anchor.anchor_name)
        return _dedupe(names)[:max_anchor_hints]


def _forward_step_source(anchor_name: str, forward_steps: list[str]) -> str:
    try:
        index = int(anchor_name.rsplit("_", 1)[-1]) - 1
    except ValueError:
        return ""
    if 0 <= index < len(forward_steps):
        return forward_steps[index]
    return ""


def _non_forward_type(anchor_name: str) -> str:
    if anchor_name.startswith("cuda_") or anchor_name.endswith("_kernel"):
        return "backend_kernel_region"
    if anchor_name in {"helpers", "user_helpers"}:
        return "helper_region"
    if anchor_name == "init_body":
        return "state_initialization"
    return "unknown"


def _anchor_priority(anchor_name: str, role: str, semantic_type: str) -> int:
    if role == "forward" and semantic_type != "unknown":
        return 5
    if anchor_name.startswith("cuda_") or anchor_name.endswith("_kernel"):
        return 4
    if role == "forward":
        return 3
    return 2


def _anchor_intent_names(semantic_type: str) -> list[str]:
    mapping = {
        "elementwise": ["fuse_elementwise_ops", "avoid_intermediate_allocations"],
        "reduction": ["preserve_reduction_semantics", "use_block_or_warp_reduction"],
        "normalization": ["fuse_statistics_and_affine", "preserve_normalization_axes"],
        "matmul": ["use_tiled_matrix_multiply", "preserve_matrix_shapes"],
        "convolution": ["preserve_layout_and_kernel_window", "reuse_input_tiles"],
        "pooling": ["preserve_pool_window", "optimize_window_traversal"],
        "attention": ["preserve_qkv_and_softmax_order", "fuse_attention_steps_when_safe"],
        "loss": ["preserve_loss_reduction", "stabilize_numeric_formula"],
        "backend_kernel_region": ["implement_backend_kernel"],
    }
    return mapping.get(semantic_type, ["inspect_source_before_editing"])


def _anchor_backend_hints(semantic_type: str, backend: str) -> list[str]:
    hints = _backend_hints_for(semantic_type).get(backend, [])
    return hints[:4]


def _intents_for(op_type: str, target_anchors: list[str], backend: str) -> list[OptimizationIntent]:
    names = _anchor_intent_names(op_type)
    hints = _backend_hints_for(op_type)
    return [
        OptimizationIntent(
            name=name,
            summary=_intent_summary(name),
            target_anchors=list(target_anchors),
            backend_hints=hints,
            risk_notes=_risk_notes_for(op_type, None),
            priority=max(1, 5 - index),
        )
        for index, name in enumerate(names)
    ]


def _backend_hints_for(op_type: str) -> dict[str, list[str]]:
    common = {
        "cuda": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"],
        "triton": ["use block programs over flattened or tiled tensors", "mask boundary elements"],
        "tilelang": ["map the computation to explicit tiles and buffers", "keep launch wiring inside kernel anchors"],
        "cute": ["describe tile layouts explicitly", "keep tensor layout assumptions local"],
    }
    if op_type == "elementwise":
        common["cuda"] += ["use one thread per element", "prefer a grid-stride loop"]
        common["triton"] += ["use tl.arange blocks over contiguous elements"]
    elif op_type == "reduction":
        common["cuda"] += ["use block or warp reductions", "preserve the reduced dimension"]
        common["triton"] += ["use tl.sum/tl.max over the block axis"]
    elif op_type == "normalization":
        common["cuda"] += ["compute statistics over the exact normalized axes", "fuse affine transform when present"]
        common["triton"] += ["keep mean/variance axes explicit"]
    elif op_type == "matmul":
        common["cuda"] += ["use tiled GEMM structure", "preserve strides and matrix dimensions"]
        common["triton"] += ["use block_m/block_n/block_k tiling"]
    elif op_type in {"convolution", "pooling"}:
        common["cuda"] += ["preserve NCHW layout unless explicitly transformed", "guard boundary indices"]
        common["triton"] += ["use masks for spatial boundaries"]
    return common


def _risk_notes_for(op_type: str, exact_facts: SemanticFactProfile | None) -> list[str]:
    mapping = {
        "elementwise": ["preserve exact activation math and broadcasting", "check fast-math numerical tolerance"],
        "reduction": ["preserve reduction dimension and keepdim behavior", "handle non-divisible sizes and boundary masks"],
        "normalization": ["preserve statistics axes, epsilon, and affine parameters", "avoid changing output dtype or shape"],
        "matmul": ["preserve M/N/K dimensions and transpose semantics", "handle tensor contiguity assumptions"],
        "convolution": ["preserve padding, stride, dilation, groups, and layout", "guard spatial boundaries"],
        "pooling": ["preserve window size, stride, padding, and ceil/count behavior", "guard boundary masks"],
        "attention": ["preserve Q/K/V order, scaling, mask, and softmax axis", "watch numerical stability"],
        "loss": ["preserve reduction mode and label semantics", "watch numerical stability"],
    }
    notes = list(mapping.get(op_type, ["inspect shape, dtype, and broadcasting assumptions before editing"]))
    if exact_facts is not None and exact_facts.kind != "unknown":
        notes.extend(exact_facts.notes[:2])
    return notes


def _summary_for(op_type: str, exact_facts: SemanticFactProfile | None) -> str:
    summaries = {
        "elementwise": "Elementwise computation; likely optimization is fusing pointwise math and avoiding intermediate tensors.",
        "reduction": "Reduction computation; preserve the reduced axis while using block/warp reductions.",
        "normalization": "Normalization computation; preserve statistics axes, epsilon, and affine state while fusing where safe.",
        "matmul": "Matrix multiplication style computation; preserve dimensions and consider tiled implementations.",
        "convolution": "Convolution style computation; preserve module state and spatial layout.",
        "pooling": "Pooling computation; preserve window traversal and boundary behavior.",
        "attention": "Attention computation; preserve Q/K/V, scaling, masks, and softmax semantics.",
        "loss": "Loss computation; preserve label semantics and reduction mode.",
    }
    summary = summaries.get(op_type, "No strong semantic pattern was recognized; inspect the forward code before editing.")
    if exact_facts is not None and exact_facts.kind != "unknown":
        summary += f" Exact operator facts available: {exact_facts.kind}."
    return summary


def _intent_summary(name: str) -> str:
    return name.replace("_", " ")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output
