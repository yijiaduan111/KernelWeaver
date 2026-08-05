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
    return f'stark_cuda_l3_p18_{digest}'

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
    def __init__(self, num_classes=1000):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
        :param num_classes: Number of output classes
        """
        self.features = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=7, stride=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        FireModule(96, 16, 64, 64),
        FireModule(128, 16, 64, 64),
        FireModule(128, 32, 128, 128),
        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        FireModule(256, 32, 128, 128),
        FireModule(256, 48, 192, 192),
        FireModule(384, 48, 192, 192),
        FireModule(384, 64, 256, 256),
        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        FireModule(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Conv2d(512, num_classes, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1))
        )
        torch.backends.cudnn.benchmark = True
        self.features = self.features.to(memory_format=torch.channels_last)
        self.classifier = self.classifier.to(memory_format=torch.channels_last)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.features(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.classifier(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return torch.flatten(x, 1)
        # <<<END_IMPROVE>>>
