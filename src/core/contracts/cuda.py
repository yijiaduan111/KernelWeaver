from __future__ import annotations

"""CUDA extension contract checks.

The checks are conservative protocol guards. They only verify that Python
extension calls, pybind exports, and C++/CUDA symbols are wired consistently.
They do not attempt to prove kernel correctness or memory safety.
"""

import re

from .base import ContractCheckResult, fail_result, pass_result

_EXTENSION_CALL_RE = re.compile(r"_stark_get_extension\s*\(\s*\)\.([A-Za-z_]\w*)\s*\(")
_PYBIND_DEF_RE = re.compile(r"m\.def\(\s*[\"']([A-Za-z_]\w*)[\"']\s*,\s*&\s*([A-Za-z_]\w*)")
_FUNCTION_SYMBOL_TEMPLATE = r"(?:Tensor|torch::Tensor|at::Tensor|void|int|float|double|auto)\s+{symbol}\s*\("


def check_cuda_contract(source_code: str) -> ContractCheckResult:
    calls = extract_extension_calls(source_code)
    if not calls:
        return pass_result()
    if not _has_extension_helper(source_code):
        return fail_result("extension_missing_helper", "candidate calls _stark_get_extension but the helper is missing")
    if _placeholder_cuda_only(source_code):
        return fail_result("extension_placeholder_source", "candidate calls extension while CUDA source is still placeholder-only")
    pybind_defs = extract_pybind_exports(source_code)
    if not pybind_defs:
        return fail_result("extension_missing_pybind_export", "candidate calls extension but PYBIND11_MODULE has no m.def exports")
    exported_names = set(pybind_defs)
    for call_name in calls:
        if call_name not in exported_names:
            return fail_result(
                "extension_entrypoint_mismatch",
                f"candidate calls extension '{call_name}' but pybind exports {sorted(exported_names)}",
            )
        target = pybind_defs[call_name]
        if not has_cpp_symbol(source_code, target):
            return fail_result(
                "extension_missing_function_symbol",
                f"pybind exports '{call_name}' via '{target}', but that function is not declared or defined",
            )
    return pass_result(f"cuda_extension_contract_ok:calls={','.join(calls)}")


def extract_extension_calls(source_code: str) -> list[str]:
    python_dispatch = _strip_cuda_source_strings(source_code)
    code_without_comments = _strip_python_comments(python_dispatch)
    return sorted(set(_EXTENSION_CALL_RE.findall(code_without_comments)))


def extract_pybind_exports(source_code: str) -> dict[str, str]:
    return {name: target for name, target in _PYBIND_DEF_RE.findall(source_code)}


def has_cpp_symbol(source_code: str, symbol: str) -> bool:
    pattern = _FUNCTION_SYMBOL_TEMPLATE.format(symbol=re.escape(symbol))
    return re.search(pattern, source_code) is not None


def _has_extension_helper(source_code: str) -> bool:
    return re.search(r"def\s+_stark_get_extension\s*\(", source_code) is not None


def _strip_cuda_source_strings(source_code: str) -> str:
    masked = re.sub(r'CUDA_CPP_SRC\s*=\s*r?""".*?"""', 'CUDA_CPP_SRC = ""', source_code, flags=re.DOTALL)
    masked = re.sub(r'CUDA_CU_SRC\s*=\s*r?""".*?"""', 'CUDA_CU_SRC = ""', masked, flags=re.DOTALL)
    return masked


def _strip_python_comments(source_code: str) -> str:
    lines = []
    for line in source_code.splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _placeholder_cuda_only(source_code: str) -> bool:
    has_placeholder = "Add CUDA kernels and exported wrapper functions here" in source_code
    if not has_placeholder:
        return False
    has_real_symbol = "__global__" in source_code or re.search(r"(?:torch::Tensor|at::Tensor)\s+[A-Za-z_]\w*\s*\(", source_code)
    return not bool(has_real_symbol)
