from __future__ import annotations

"""CUDA extension contract checks.

The checks are conservative protocol guards. They only verify that Python
extension calls, pybind exports, and C++/CUDA symbols are wired consistently.
They do not attempt to prove kernel correctness or memory safety.
"""

import ast
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
    loader_contract = _check_loader_contract(source_code)
    if not loader_contract.ok:
        return loader_contract
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


def _check_loader_contract(source_code: str) -> ContractCheckResult:
    helper = _python_function_node(source_code, "_stark_get_extension")
    if helper is None:
        return fail_result("extension_missing_helper", "candidate calls _stark_get_extension but the helper is missing")
    helper_body = ast.unparse(helper) if hasattr(ast, "unparse") else "\n".join(ast.dump(item) for item in helper.body)
    helper_calls = _called_names(helper)
    if "load_inline" not in helper_calls and "load_inline" not in helper_body:
        return fail_result("extension_invalid_load_inline_contract", "_stark_get_extension does not call load_inline")
    if "_stark_strip_anchor_markers" in helper_calls:
        strip_helper = _python_function_node(source_code, "_stark_strip_anchor_markers")
        if strip_helper is None:
            return fail_result("extension_missing_strip_helper", "_stark_get_extension uses _stark_strip_anchor_markers but that helper is missing")
        if not _valid_strip_helper(strip_helper):
            return fail_result("extension_invalid_strip_helper", "_stark_strip_anchor_markers no longer preserves non-anchor source lines")
    if "_stark_extension_name" in helper_calls:
        name_helper = _python_function_node(source_code, "_stark_extension_name")
        if name_helper is None:
            return fail_result("extension_missing_name_helper", "_stark_get_extension uses _stark_extension_name but that helper is missing")
        if not _valid_name_helper(name_helper):
            return fail_result("extension_invalid_name_helper", "_stark_extension_name no longer hashes CUDA_CPP_SRC and CUDA_CU_SRC")
    if "cpp_sources" not in helper_body or "CUDA_CPP_SRC" not in helper_body:
        return fail_result("extension_invalid_load_inline_contract", "load_inline must receive CUDA_CPP_SRC as cpp_sources")
    if "cuda_sources" not in helper_body or "CUDA_CU_SRC" not in helper_body:
        return fail_result("extension_invalid_load_inline_contract", "load_inline must receive CUDA_CU_SRC as cuda_sources")
    if "with_cuda" not in helper_body or "True" not in helper_body:
        return fail_result("extension_invalid_load_inline_contract", "load_inline must set with_cuda=True")
    return pass_result()


def _python_function_node(source_code: str, name: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names



def _valid_strip_helper(node: ast.FunctionDef) -> bool:
    text = ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)
    required = ["source.splitlines", "cleaned_lines.append(line)", "return '\\n'.join(cleaned_lines)"]
    if not all(item in text for item in required):
        return False
    if "<<<IMPROVE:" not in text or "<<<END_IMPROVE>>>" not in text:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.For):
            for index, stmt in enumerate(child.body[:-1]):
                if isinstance(stmt, ast.Continue) and _block_contains_append(child.body[index + 1:]):
                    return False
    return True


def _valid_name_helper(node: ast.FunctionDef) -> bool:
    text = ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)
    return "hashlib.sha1" in text and "CUDA_CPP_SRC" in text and "CUDA_CU_SRC" in text


def _block_contains_append(statements: list[ast.stmt]) -> bool:
    for stmt in statements:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "append":
                return True
    return False
