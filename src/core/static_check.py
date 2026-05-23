"""Cheap static guards for generated candidates.

These checks catch protocol and CUDA extension interface mistakes before
launching KernelBench evaluation. They are intentionally conservative and do not
try to prove semantic correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..utils import extract_anchor_names


@dataclass
class StaticCheckResult:
    ok: bool
    failure_type: str | None = None
    logs: list[str] = field(default_factory=list)


def check_candidate_static(source_code: str, backend: str | None = None) -> StaticCheckResult:
    checks = [_check_basic_source(source_code)]
    if backend == "cuda" or "CUDA_CU_SRC" in source_code or "_stark_get_extension()" in source_code:
        checks.append(_check_cuda_extension_contract(source_code))
    logs: list[str] = []
    for result in checks:
        logs.extend(result.logs)
        if not result.ok:
            return StaticCheckResult(False, result.failure_type, logs)
    return StaticCheckResult(True, logs=logs)


def _check_basic_source(source_code: str) -> StaticCheckResult:
    stripped = source_code.lstrip()
    if stripped.startswith("{") or "\"anchor_patches\"" in source_code or "'anchor_patches'" in source_code:
        return _fail("unapplied_patch_payload", "candidate still contains an unapplied anchor_patches payload")
    if "class ModelNew" not in source_code:
        return _fail("missing_modelnew", "candidate is missing class ModelNew")
    if "def forward" not in source_code:
        return _fail("missing_forward", "candidate is missing forward method")
    if not extract_anchor_names(source_code):
        return _fail("no_anchor_markers", "candidate has no anchor markers")
    return StaticCheckResult(True)


def _check_cuda_extension_contract(source_code: str) -> StaticCheckResult:
    code_without_comments = _strip_python_comments(source_code)
    extension_calls = sorted(set(re.findall(r"_stark_get_extension\(\)\.([A-Za-z_]\w*)\s*\(", code_without_comments)))
    if not extension_calls:
        return StaticCheckResult(True)

    pybind_defs = {name: target for name, target in re.findall(r"m\.def\(\s*[\"']([A-Za-z_]\w*)[\"']\s*,\s*&\s*([A-Za-z_]\w*)", source_code)}
    if not pybind_defs:
        return _fail("missing_pybind_binding", "forward calls extension but PYBIND11_MODULE has no m.def binding")

    for call_name in extension_calls:
        if call_name not in pybind_defs:
            return _fail("extension_entrypoint_mismatch", f"forward calls extension '{call_name}' but pybind exports {sorted(pybind_defs)}")
        target = pybind_defs[call_name]
        if not _has_cpp_symbol(source_code, target):
            return _fail("missing_bound_function", f"pybind exports '{call_name}' via '{target}', but that function is not declared or defined")

    if _placeholder_cuda_only(source_code):
        return _fail("placeholder_cuda_extension", "forward calls extension while CUDA source still appears to be placeholder-only")
    return StaticCheckResult(True)


def _strip_python_comments(source_code: str) -> str:
    lines = []
    for line in source_code.splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _has_cpp_symbol(source_code: str, symbol: str) -> bool:
    pattern = rf"(?:Tensor|torch::Tensor|void|int|float|double|auto)\s+{re.escape(symbol)}\s*\("
    return re.search(pattern, source_code) is not None


def _placeholder_cuda_only(source_code: str) -> bool:
    if "_stark_get_extension()." not in source_code:
        return False
    has_kernel = "__global__" in source_code or "torch::Tensor" in source_code or "at::Tensor" in source_code
    has_placeholder = "Add CUDA kernels and exported wrapper functions here" in source_code
    return has_placeholder and not has_kernel


def _fail(failure_type: str, message: str) -> StaticCheckResult:
    return StaticCheckResult(False, failure_type, [message])
