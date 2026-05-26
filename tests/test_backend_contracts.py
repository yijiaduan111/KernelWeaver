from src.core.contracts import check_backend_contract


CUDA_BASE = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# <<<IMPROVE:helpers>>>
def _stark_get_extension():
    return load_inline(name="x", cpp_sources=CUDA_CPP_SRC, cuda_sources=CUDA_CU_SRC, functions=None, with_cuda=True)
# <<<END_IMPROVE>>>

CUDA_CPP_SRC = r"""
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("foo", &foo, "foo"); }
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
torch::Tensor foo(torch::Tensor x) { return x; }
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        return _stark_get_extension().foo(x)
        # <<<END_IMPROVE>>>
'''


def test_cuda_contract_accepts_matching_extension():
    result = check_backend_contract(CUDA_BASE, backend="cuda")
    assert result.ok
    assert any("cuda_extension_contract_ok" in log for log in result.logs)


def test_cuda_contract_rejects_missing_helper():
    code = CUDA_BASE.replace("def _stark_get_extension():", "def _broken_extension():")
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_missing_helper"


def test_cuda_contract_rejects_missing_pybind_export():
    code = CUDA_BASE.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("foo", &foo, "foo"); }', 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}')
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_missing_pybind_export"


def test_cuda_contract_rejects_entrypoint_mismatch():
    code = CUDA_BASE.replace('return _stark_get_extension().foo(x)', 'return _stark_get_extension().bar(x)')
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_entrypoint_mismatch"


def test_cuda_contract_rejects_missing_function_symbol():
    code = CUDA_BASE.replace('torch::Tensor foo(torch::Tensor x) { return x; }', '// no foo symbol here')
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_missing_function_symbol"


def test_cuda_contract_rejects_placeholder_source_when_extension_called():
    code = CUDA_BASE.replace('torch::Tensor foo(torch::Tensor x) { return x; }', '// Add CUDA kernels and exported wrapper functions here.')
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_placeholder_source"


def test_cuda_contract_allows_torch_fallback_without_extension_call():
    code = CUDA_BASE.replace('return _stark_get_extension().foo(x)', 'return x')
    result = check_backend_contract(code, backend="cuda")
    assert result.ok


def test_non_cuda_backends_skip_cuda_contract():
    broken_cuda = CUDA_BASE.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("foo", &foo, "foo"); }', 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}')
    for backend in ["triton", "tilelang", "cute"]:
        result = check_backend_contract(broken_cuda, backend=backend)
        assert result.ok
        assert any("backend_contract_skipped" in log for log in result.logs)


LOADER_STYLE_CUDA = CUDA_BASE.replace(
    'def _stark_get_extension():\n    return load_inline(name="x", cpp_sources=CUDA_CPP_SRC, cuda_sources=CUDA_CU_SRC, functions=None, with_cuda=True)',
    "def _stark_strip_anchor_markers(source: str) -> str:\n    cleaned_lines = []\n    for line in source.splitlines():\n        stripped = line.lstrip()\n        if stripped.startswith('# <<<IMPROVE:') or stripped.startswith('# <<<END_IMPROVE>>>'):\n            continue\n        cleaned_lines.append(line)\n    return '\\n'.join(cleaned_lines)\n\ndef _stark_extension_name() -> str:\n    digest = hashlib.sha1((_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode('utf-8')).hexdigest()[:12]\n    return f'x_{digest}'\n\ndef _stark_get_extension():\n    return load_inline(name=_stark_extension_name(), cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC), cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC), functions=None, with_cuda=True)",
)


def test_cuda_contract_rejects_missing_strip_helper_when_loader_uses_it():
    code = LOADER_STYLE_CUDA.replace('def _stark_strip_anchor_markers(source: str) -> str:', 'def _broken_strip_helper(source: str) -> str:')
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_missing_strip_helper"


def test_cuda_contract_rejects_missing_name_helper_when_loader_uses_it():
    code = LOADER_STYLE_CUDA.replace('def _stark_extension_name() -> str:', 'def _broken_extension_name() -> str:')
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_missing_name_helper"


def test_cuda_contract_rejects_missing_cuda_sources_argument():
    code = CUDA_BASE.replace('cuda_sources=CUDA_CU_SRC, ', '')
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_invalid_load_inline_contract"


def test_cuda_contract_rejects_invalid_strip_helper():
    code = LOADER_STYLE_CUDA.replace(
        "if stripped.startswith('# <<<IMPROVE:') or stripped.startswith('# <<<END_IMPROVE>>>'):\n            continue",
        "continue",
    )
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_invalid_strip_helper"


def test_cuda_contract_rejects_invalid_name_helper():
    code = LOADER_STYLE_CUDA.replace("digest = hashlib.sha1((_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode('utf-8')).hexdigest()[:12]", "digest = 'fixed'")
    result = check_backend_contract(code, backend="cuda")
    assert not result.ok
    assert result.failure_type == "extension_invalid_name_helper"
