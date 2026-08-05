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
    return f'stark_cuda_l3_p27_{digest}'

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
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                :param input_channels: int, Number of input channels for the first layer
                :param stages: int, Number of stages in the RegNet architecture
                :param block_widths: List[int], Width (number of channels) for each block in the stages
                :param output_classes: int, Number of output classes for classification
                """
        self.stages = stages
        self.block_widths = block_widths
        layers = []
        current_channels = input_channels
        for i in range(stages):
                    layers.append(self._make_stage(current_channels, block_widths[i]))
                    current_channels = block_widths[i]
        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(block_widths[-1], output_classes)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass through the RegNet model.
                :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
                :return: torch.Tensor of shape (batch_size, output_classes)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.feature_extractor(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = torch.mean(x, dim=[2, 3])
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = self.fc(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
