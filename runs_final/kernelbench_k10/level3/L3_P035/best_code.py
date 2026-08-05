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
    return f'stark_cuda_l3_p35_{digest}'

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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initialize the LSTM model.

                :param input_size: The number of expected features in the input `x`
                :param hidden_size: The number of features in the hidden state `h`
                :param num_layers: Number of recurrent layers
                :param output_size: The number of output features
                :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
                """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        # <<<END_IMPROVE>>>

    def forward(self, x, h0=None, c0=None):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass through the LSTM model.

                :param x: The input tensor, shape (batch_size, sequence_length, input_size)
                :param h0: Optional initial hidden state (num_layers, batch_size, hidden_size)
                :param c0: Optional initial cell state (num_layers, batch_size, hidden_size)
                :return: The output tensor, shape (batch_size, output_size)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        batch_size = x.size(0)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if h0 is None:
                    h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if c0 is None:
                    c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        out, _ = self.lstm(x, (h0, c0))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        out = self.fc(out[:, -1, :])
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return out
        # <<<END_IMPROVE>>>
