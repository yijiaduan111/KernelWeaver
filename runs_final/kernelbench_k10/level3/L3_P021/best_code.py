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
    return f'stark_cuda_l3_p21_{digest}'

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
def _stark_fold_conv_bn_params(conv, bn):
    """
    Fold BatchNorm2d into the preceding Conv2d for inference.
    Returns (folded_weight, folded_bias) ready for F.conv2d.
    """
    with torch.no_grad():
        scale = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
        # Reshape scale to [out_channels, 1, 1, 1] for broadcast over conv weight
        scale_w = scale.view(-1, 1, 1, 1)
        folded_weight = conv.weight * scale_w
        if conv.bias is None:
            folded_bias = bn.bias - bn.running_mean * scale
        else:
            folded_bias = (conv.bias - bn.running_mean) * scale + bn.bias
        return folded_weight, folded_bias
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
            self.expand_act = nn.ReLU6(inplace=True)
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        self.depthwise_act = nn.ReLU6(inplace=True)
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the MBConv block.

                :param x: The input tensor, shape (batch_size, in_channels, H, W)
                :return: The output tensor, shape (batch_size, out_channels, H', W')
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        identity = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if hasattr(self, 'expand_conv'):
            if (not self.training) and x.is_cuda:
                weight, bias = _stark_fold_conv_bn_params(self.expand_conv, self.expand_bn)
                x = F.conv2d(x, weight, bias,
                             stride=self.expand_conv.stride,
                             padding=self.expand_conv.padding,
                             dilation=self.expand_conv.dilation,
                             groups=self.expand_conv.groups)
                x = self.expand_act(x)
            else:
                x = self.expand_act(self.expand_bn(self.expand_conv(x)))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if (not self.training) and x.is_cuda:
            weight, bias = _stark_fold_conv_bn_params(self.depthwise_conv, self.depthwise_bn)
            x = F.conv2d(x, weight, bias,
                         stride=self.depthwise_conv.stride,
                         padding=self.depthwise_conv.padding,
                         dilation=self.depthwise_conv.dilation,
                         groups=self.depthwise_conv.groups)
            x = self.depthwise_act(x)
        else:
            x = self.depthwise_act(self.depthwise_bn(self.depthwise_conv(x)))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if (not self.training) and x.is_cuda:
            weight, bias = _stark_fold_conv_bn_params(self.project_conv, self.project_bn)
            x = F.conv2d(x, weight, bias,
                         stride=self.project_conv.stride,
                         padding=self.project_conv.padding,
                         dilation=self.project_conv.dilation,
                         groups=self.project_conv.groups)
        else:
            x = self.project_bn(self.project_conv(x))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        if self.use_residual:
                    x += identity
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
