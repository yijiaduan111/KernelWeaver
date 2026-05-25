import json

from src.core.candidate import normalize_candidate
from src.core.static_check import check_candidate_static
from src.core.tree import TreeMemory
from src.models import EvaluationResult, PlanProposal, SearchNode, StarkConfig


PARENT = '''
import torch
import torch.nn as nn

# <<<IMPROVE:helpers>>>
# helper
# <<<END_IMPROVE>>>
CUDA_CPP_SRC = r"""
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
# <<<END_IMPROVE>>>
"""
CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
# <<<END_IMPROVE>>>
"""
class ModelNew(nn.Module):
    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        return x
        # <<<END_IMPROVE>>>
'''


def test_normalize_applies_anchor_patch():
    raw = '{"anchor_patches":[{"anchor_name":"forward_stmt_1","operation":"replace","body":"return x + 1"}]}'
    result = normalize_candidate(PARENT, raw)
    assert result.ok
    assert "return x + 1" in result.code
    assert "anchor_patches" not in result.code


def test_normalize_rejects_bad_patch_payload():
    result = normalize_candidate(PARENT, '{"anchor_patches":')
    assert not result.ok
    assert result.failure_type == "invalid_patch_format"


def test_normalize_applies_region_patch_and_preserves_scaffold():
    raw = '{"region_patches":[{"region":"forward_stmt_1","operation":"replace","body":"return x + 2"}]}'
    result = normalize_candidate(PARENT, raw)
    assert result.ok
    assert "return x + 2" in result.code
    assert "region_patches" not in result.code
    assert "# <<<IMPROVE:forward_stmt_1>>>" in result.code
    assert "# <<<END_IMPROVE>>>" in result.code


def test_normalize_rejects_unknown_region_patch():
    raw = '{"region_patches":[{"region":"missing_region","operation":"replace","body":"return x"}]}'
    result = normalize_candidate(PARENT, raw)
    assert not result.ok
    assert result.failure_type == "region_apply_failed"


def test_static_check_rejects_unapplied_region_payload():
    result = check_candidate_static('{"region_patches": []}', backend="cuda")
    assert not result.ok
    assert result.failure_type == "unapplied_patch_payload"


def test_static_check_rejects_extension_mismatch():
    code = PARENT.replace('return x', 'return _stark_get_extension().missing(x)')
    result = check_candidate_static(code, backend="cuda")
    assert not result.ok
    assert result.failure_type in {"missing_pybind_binding", "extension_entrypoint_mismatch"}


def test_static_check_accepts_matching_extension_contract():
    code = PARENT.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}', 'torch::Tensor foo(torch::Tensor x);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("foo", &foo, "foo"); }')
    code = code.replace('#include <torch/extension.h>\n# <<<END_IMPROVE>>>', '#include <torch/extension.h>\ntorch::Tensor foo(torch::Tensor x) { return x; }\n# <<<END_IMPROVE>>>')
    code = code.replace('return x', 'return _stark_get_extension().foo(x)')
    result = check_candidate_static(code, backend="cuda")
    assert result.ok


def test_tree_excludes_no_anchor_nodes():
    root = SearchNode(node_id="root", parent_id=None, depth=0, code=PARENT, origin="root", compile_ok=True, correct=True, score=1.0)
    tree = TreeMemory(root, StarkConfig())
    proposal = PlanProposal("s", "s", [], "gain")
    eval_result = EvaluationResult(False, False, None, float("inf"), failure_type="broken_anchor_markers", failure_stage="compile")
    child = tree.add_child("root", "class ModelNew: pass", proposal, eval_result, "plan_code")
    assert tree.exclusion_reason(child.node_id, StarkConfig()) == "no_anchor_unexpandable"


def test_region_patch_normalizes_forward_indent_and_is_cuda_property():
    raw = json.dumps({
        "region_patches": [
            {
                "region": "forward_stmt_1",
                "operation": "replace",
                "body": "if x.is_cuda() and x.dtype == torch.float32:\n            return x + 1\n        return x",
            }
        ]
    })
    result = normalize_candidate(PARENT, raw)
    assert result.ok
    assert "if x.is_cuda and x.dtype == torch.float32:" in result.code
    assert "x.is_cuda()" not in result.code
    assert "    return x + 1" in result.code
    assert "        return x + 1" in result.code
    assert check_candidate_static(result.code, backend="cuda").ok


def test_region_patch_does_not_rewrite_cuda_body_api_names():
    raw = '{"region_patches":[{"region":"cuda_cu","operation":"replace","body":"// x.is_cuda() inside CUDA string should stay text"}]}'
    result = normalize_candidate(PARENT, raw)
    assert result.ok
    assert "x.is_cuda()" in result.code


def test_static_check_reports_python_region_syntax_error():
    bad = PARENT.replace("return x", "if True:\nreturn x")
    result = check_candidate_static(bad, backend="cuda")
    assert not result.ok
    assert result.failure_type == "python_region_syntax_error"


def test_tree_throttles_compile_failure_debug_and_ignores_bad_root_children_for_throttle():
    root = SearchNode(node_id="root", parent_id=None, depth=0, code=PARENT, origin="root", compile_ok=True, correct=True, score=1.0)
    config = StarkConfig(root_child_limit=1, debug_retry_limit=3)
    tree = TreeMemory(root, config)
    proposal = PlanProposal("s", "s", [], "gain")
    compile_eval = EvaluationResult(False, False, None, float("inf"), failure_type="python_region_syntax_error", failure_stage="compile")
    child = tree.add_child("root", PARENT, proposal, compile_eval, "plan_code")
    assert tree.exclusion_reason("root", config) is None
    assert tree.exclusion_reason(child.node_id, config) is None
    child.debug_attempts = 1
    assert tree.exclusion_reason(child.node_id, config) == "compile_failure_debug_throttled"
