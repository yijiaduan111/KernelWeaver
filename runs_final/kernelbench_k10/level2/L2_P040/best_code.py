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
    return f'stark_cuda_l2_p40_{digest}'

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
        A model that performs a matrix multiplication, scaling, and residual addition.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            scaling_factor (float): Scaling factor to apply after matrix multiplication.
        """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._alpha = float(1.0 + scaling_factor)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the model.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_features).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_features).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        alpha = 1.0 + self.scaling_factor
        if self.matmul.bias is not None:
            return torch.addmm(self.matmul.bias * alpha, x, self.matmul.weight.t(), beta=1.0, alpha=alpha)
        return torch.matmul(x, self.matmul.weight.t()) * alpha
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        pass
        # <<<END_IMPROVE>>>
