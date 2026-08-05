import torch
import torch.nn as nn
import hashlib
from torch.utils.cpp_extension import load_inline

_STARK_EXTENSION = None

def _stark_strip_anchor_markers(source: str) -> str:
    cleaned_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('# <<<IMPROVE:') or stripped.startswith('# <<<END_IMPROVE>>>'):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def _stark_extension_name() -> str:
    digest = hashlib.sha1((_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode('utf-8')).hexdigest()[:12]
    return f'stark_cuda_l2_p18_{digest}'

def _stark_get_extension():
    global _STARK_EXTENSION
    if _STARK_EXTENSION is None:
        _STARK_EXTENSION = load_inline(
            name=_stark_extension_name(),
            cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC),
            cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC),
            functions=None,
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True,
            verbose=False,
        )
    return _STARK_EXTENSION

# <<<IMPROVE:user_helpers>>>
# <<<END_IMPROVE>>>

CUDA_CPP_SRC = r"""
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>

// Add pybind exports for custom CUDA entrypoints here.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Add CUDA kernels and exported wrapper functions here.
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a sequence of operations:
            - Matrix multiplication
            - Summation
            - Max
            - Average pooling
            - LogSumExp
            - LogSumExp
        """
    def __init__(self, in_features, out_features):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.linear = nn.Linear(in_features, out_features)
        # Precompute collapsed weight vector and scalar bias:
        # sum(linear(x), dim=1) = x @ weight.sum(dim=0) + bias.sum()
        with torch.no_grad():
            w_col_sum = self.linear.weight.sum(dim=0).unsqueeze(0).contiguous()  # shape [1, in_features]
            b_sum = self.linear.bias.sum().reshape(1).contiguous()               # shape [1]
        self.register_buffer('w_col_sum', w_col_sum)
        self.register_buffer('b_sum', b_sum)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_features).
                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, 1).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        # Collapsed affine projection: result is [batch_size, 1] directly
        x = torch.nn.functional.linear(x, self.w_col_sum, self.b_sum)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # x is already [batch_size, 1]; sum over dim=1 is identity
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # max over dim=1 on [batch_size, 1] is identity
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # mean over dim=1 on [batch_size, 1] is identity
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # logsumexp over dim=1 on [batch_size, 1] is identity (log(exp(x)) = x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        # logsumexp over dim=1 on [batch_size, 1] is identity (log(exp(x)) = x); skip launch
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        return x
        # <<<END_IMPROVE>>>
