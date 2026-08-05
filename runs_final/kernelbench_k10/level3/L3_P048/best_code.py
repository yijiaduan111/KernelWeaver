import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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
    return f'stark_cuda_l3_p48_{digest}'

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
def _stark_modelnew_segsum(self, x):
    T = x.size(-1)
    prefix = F.pad(torch.cumsum(x, dim=-1), (1, 0))
    x = prefix[..., 1:].unsqueeze(-1) - prefix[..., 1:].unsqueeze(-2)
    mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=0)
    x = x.masked_fill(~mask, float('-inf'))
    return x

nn.Module.segsum = _stark_modelnew_segsum
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
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Mamba Structured State Space model implementation for benchmarking.

                :param batch_size: Size of the batch
                :param seq_length: Length of the input sequence
                :param n_heads: Number of attention heads
                :param d_head: Dimension of each head
                :param d_state: Dimension of the state space
                :param block_len: Length of each block for chunked computation
                """
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        # <<<END_IMPROVE>>>

    def forward(self, X, initial_states=None):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass implementing the SSD operation.

                :param X: Input tensor of shape (batch, length, n_heads, d_head)
                :param initial_states: Optional initial states
                :return: Output tensor Y and final state
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        X_blocks, A_blocks, B_blocks, C_blocks = [
                    rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
                    for x in (X, self.A, self.B, self.C)
                ]
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        L = torch.exp(self.segsum(A_blocks))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                                     C_blocks, B_blocks, L, X_blocks)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                                    B_blocks, decay_states, X_blocks)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        if initial_states is None:
                    initial_states = torch.zeros_like(states[:, :1])
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_10>>>
        states = torch.cat([initial_states, states], dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_11>>>
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_12>>>
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_13>>>
        states = new_states[:, :-1]
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_14>>>
        state_decay_out = torch.exp(A_cumsum)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_15>>>
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                                   C_blocks, states, state_decay_out)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_16>>>
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_17>>>
        return Y
        # <<<END_IMPROVE>>>
