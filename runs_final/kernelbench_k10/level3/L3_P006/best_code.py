import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return f'stark_cuda_l3_p6_{digest}'

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
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                :param in_channels: Number of input channels
                :param out_1x1: Number of output channels for the 1x1 convolution
                :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
                :param out_3x3: Number of output channels for the 3x3 convolution
                :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
                :param out_5x5: Number of output channels for the 5x5 convolution
                :param pool_proj: Number of output channels for the pooling projection
                """
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
                    nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
                    nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
                )
        self.branch5x5 = nn.Sequential(
                    nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
                    nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
                )
        self.branch_pool = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_channels, pool_proj, kernel_size=1)
                )
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                :param x: Input tensor, shape (batch_size, in_channels, height, width)
                :return: Output tensor, shape (batch_size, out_channels, height, width)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        branch1x1 = self.branch1x1(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        branch3x3 = self.branch3x3(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        branch5x5 = self.branch5x5(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        branch_pool = self.branch_pool(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return torch.cat(outputs, 1)
        # <<<END_IMPROVE>>>
