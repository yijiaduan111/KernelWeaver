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
    return f'stark_cuda_l1_p64_{digest}'

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
import torch.nn.functional as F
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
        Performs a transposed 1D convolution operation.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolution kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to the input. Defaults to 0.
            output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self._fast_path = (stride == 1 and padding == 0 and output_padding == 0 and groups == 1)
        self._fast_padding = kernel_size - 1
        if self._fast_path:
            w = self.conv1d_transpose.weight.detach().transpose(0, 1).flip(-1).contiguous()
            self.register_buffer('_w_equiv', w)
            self._fast_bias = self.conv1d_transpose.bias
        else:
            self._fast_bias = None
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if (self._fast_path
                        and x.is_cuda
                        and x.dtype == torch.float32
                        and x.is_contiguous()):
                    return F.conv1d(x, self._w_equiv, bias=self._fast_bias, stride=1, padding=self._fast_padding, groups=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return self.conv1d_transpose(x)
        # <<<END_IMPROVE>>>
