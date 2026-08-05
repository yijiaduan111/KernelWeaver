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
    return f'stark_cuda_l3_p7_{digest}'

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
        torch.backends.cudnn.benchmark = True
        self.to(memory_format=torch.channels_last)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                :param x: Input tensor, shape (batch_size, 3, height, width)
                :return: Output tensor, shape (batch_size, num_classes)
                """
        x = x.to(memory_format=torch.channels_last)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.maxpool1(F.relu(self.conv1(x)))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = F.relu(self.conv2(x))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = self.maxpool2(F.relu(self.conv3(x)))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = self.inception3a(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        x = self.inception3b(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        x = self.maxpool3(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        x = self.inception4a(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        x = self.inception4b(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_10>>>
        x = self.inception4c(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_11>>>
        x = self.inception4d(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_12>>>
        x = self.inception4e(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_13>>>
        x = self.maxpool4(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_14>>>
        x = self.inception5a(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_15>>>
        x = self.inception5b(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_16>>>
        x = self.avgpool(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_17>>>
        x = torch.flatten(x, 1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_18>>>
        x = self.dropout(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_19>>>
        x = self.fc(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_20>>>
        return x
        # <<<END_IMPROVE>>>
