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
    return f'stark_cuda_l3_p34_{digest}'

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
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initialize the Vanilla RNN model.

                :param input_size: The number of input features (int).
                :param hidden_size: The size of the hidden state (int).
                :param output_size: The number of output features (int).
                """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the Vanilla RNN.

                :param x: Input tensor of shape (seq_len, batch_size, input_size)
                :param h0: Initial hidden state tensor of shape (batch_size, hidden_size)
                :return: Output tensor of shape (seq_len, batch_size, output_size)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        seq_len, batch_size, _ = x.size()
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        hidden = h0.to(x.device)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # Precompute full-sequence input projection once outside the loop.
        weight_ih = self.i2h.weight[:, :self.input_size]
        weight_hh = self.i2h.weight[:, self.input_size:]
        bias_h = self.i2h.bias

        flat_x = x.contiguous().view(seq_len * batch_size, self.input_size)
        x_proj = flat_x.mm(weight_ih.t()).view(seq_len, batch_size, self.hidden_size)
        x_proj_biased = x_proj + bias_h
        weight_hh_t = weight_hh.t().contiguous()
        hidden_states = x.new_empty((seq_len, batch_size, self.hidden_size))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        for t in range(seq_len):
            hidden = torch.tanh(torch.addmm(x_proj_biased[t], hidden, weight_hh_t))
            hidden_states[t] = hidden
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        flat_hidden = hidden_states.view(seq_len * batch_size, self.hidden_size)
        flat_out = flat_hidden.mm(self.h2o.weight.t()) + self.h2o.bias
        return flat_out.view(seq_len, batch_size, self.output_size)
        # <<<END_IMPROVE>>>
