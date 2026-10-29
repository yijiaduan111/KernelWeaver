"""Bounded parameter mutations; no model calls and no correctness shortcuts.

The search layer consumes backend-neutral proposals. Only the CUDA extractor
knows about launch syntax. Unsupported/ambiguous code deliberately yields none.
These syntactic guards are NOT an equivalence proof: every variant is evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re

from ..models import AnchorEdit, PlanProposal
from ..utils import _ANCHOR_PATTERN, normalized_code_hash, preserve_anchor_scaffold


@dataclass(frozen=True)
class ParameterVariant:
    code: str
    anchor: str
    parameter: str
    original: int
    value: int
    backend: str
    constraints: tuple[str, ...]

    def metadata(self) -> dict:
        result = {k: v for k, v in asdict(self).items() if k != "code"}
        result["constraints"] = list(self.constraints)
        return result

    def proposal(self) -> PlanProposal:
        instruction = f"Change only {self.parameter}: {self.original} -> {self.value}."
        return PlanProposal(
            strategy_name=f"local_tune:{self.parameter}={self.value}",
            strategy_summary=instruction,
            anchor_edits=[AnchorEdit(self.anchor, instruction)],
            expected_gain="Empirically test launch granularity without rewriting the algorithm.",
            risk_notes="Syntactic guards only; full correctness and timing checks are mandatory.",
        )


def parameter_variants(code: str, backend: str | None) -> list[ParameterVariant]:
    """Small allowlisted space, not arbitrary regex replacement of constants."""
    if backend != "cuda":
        return []
    return _cuda_launch_variants(code)


def _cuda_launch_variants(code: str) -> list[ParameterVariant]:
    regions = [m for m in _ANCHOR_PATTERN.finditer(code) if m.group("name") == "cuda_cu"]
    if len(regions) != 1:
        return []
    region = regions[0]
    body = region.group("body")
    # Mask C/C++ comments, preserving character offsets for one-literal patches.
    masked = re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: " " * len(m[0]), body, flags=re.S)
    if not all(token in masked for token in ("__global__", "blockDim.x", "threadIdx.x")):
        return []
    # Fixed shared arrays, compile-time launch contracts, multidimensional
    # mappings and cooperative groups need a richer dependency analysis.
    if re.search(r"(?:blockDim|threadIdx)\.[yz]|__launch_bounds__|cooperative_groups|__cluster_dims__", masked):
        return []
    if re.search(r"__shared__[^;]*\[\s*[^\s\]]", masked):
        return []
    declarations = list(re.finditer(
        r"\b(?:(?:const|constexpr)\s+)?(?:unsigned\s+)?(?:int|int32_t|int64_t)\s+"
        r"(?P<name>threads|threads_per_block|block_size|blockSize|BLOCK_SIZE)\s*=\s*(?P<value>\d+)\s*;",
        masked,
    ))
    variants = []
    for declaration in declarations:
        name = declaration.group("name")
        original = int(declaration.group("value"))
        if original not in (64, 128, 256, 512):
            continue
        # Duplicate/shadowed declarations or reassignment make the patch ambiguous.
        assignments = re.findall(rf"\b{re.escape(name)}\s*=(?!=)", masked)
        if len(assignments) != 1 or re.search(rf"\b{name}\s*(?:\+=|-=|\*=|/=|\+\+|--)", masked):
            continue
        if not re.search(rf"<<<\s*[^,;]+,\s*{name}\s*(?:,|>>>)", masked[declaration.end():]):
            continue
        start = region.start("body") + declaration.start("value")
        end = region.start("body") + declaration.end("value")
        # At most two adjacent powers of two. Keep warp alignment and <=512.
        for value in (original // 2, original * 2):
            if value not in (64, 128, 256, 512):
                continue
            candidate = code[:start] + str(value) + code[end:]
            if not preserve_anchor_scaffold(code, candidate):
                continue
            variants.append(ParameterVariant(
                candidate, "cuda_cu", name, original, value, "cuda",
                ("one_integer_literal", "one_dimensional_launch", "warp_aligned_power_of_two",
                 "no_fixed_shared_arrays", "full_evaluation_required"),
            ))
    return variants


def choose_local_variant(tree, backend: str | None):
    """Tune validated generated kernels only, never root or failed/tuned nodes."""
    seen = {normalized_code_hash(n.code) for n in tree.nodes.values()}
    parents = sorted(
        (n for n in tree.nodes.values() if n.node_id != tree.root_id and n.correct
         and n.compile_ok and n.origin != "local_tune" and not n.is_failure
         and n.node_id not in tree.pruned_nodes
         and n.runtime is not None),
        key=lambda n: (n.runtime, n.node_id),
    )
    for parent in parents:
        for variant in parameter_variants(parent.code, backend):
            if normalized_code_hash(variant.code) not in seen:
                return parent, variant
    return None
