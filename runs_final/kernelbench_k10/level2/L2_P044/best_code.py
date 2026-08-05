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
    return f'stark_cuda_l2_p44_{digest}'

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
        Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
        another global average pooling
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        x = x.to(memory_format=torch.channels_last)
        x = torch.nn.functional.conv_transpose2d(x, self.conv_transpose.weight, bias=None, stride=self.conv_transpose.stride, padding=self.conv_transpose.padding, output_padding=self.conv_transpose.output_padding, groups=self.conv_transpose.groups, dilation=self.conv_transpose.dilation)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if self.conv_transpose.bias is not None:
            x = x + self.conv_transpose.bias.view(1, -1, 1, 1)
        x = x * self.multiplier
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
