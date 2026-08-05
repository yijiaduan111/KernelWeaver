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
    return f'stark_cuda_l3_p49_{digest}'

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
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        # <<<IMPROVE:init_body>>>
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

        chunk_count = seq_length // block_len

        def _forward_final_state(X, initial_states=None):
            B_sz = X.shape[0]

            X_blocks = X.view(B_sz, chunk_count, block_len, n_heads, d_head)
            A_blocks = self.A.view(B_sz, chunk_count, block_len, n_heads).permute(0, 3, 1, 2).contiguous()
            B_blocks = self.B.view(B_sz, chunk_count, block_len, n_heads, d_state)

            A_cumsum = torch.cumsum(A_blocks, dim=-1)
            decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)

            chunk_prefix = torch.cumsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), dim=-1)
            final_decay = torch.exp(chunk_prefix[:, :, -1:] - chunk_prefix[:, :, 1:])
            result = torch.einsum("bhc,bchpn->bhpn", final_decay, states)

            if initial_states is not None:
                result = result + torch.exp(chunk_prefix[:, :, -1])[:, :, None, None] * initial_states[:, 0]

            return result

        self.forward = _forward_final_state
        # <<<END_IMPROVE>>>

    def forward(self, X, initial_states=None):
        # <<<IMPROVE:forward_stmt_1>>>
        X_blocks = rearrange(X, "b (c l) ... -> b c l ...", l=self.block_len)
        A_blocks = rearrange(self.A, "b (c l) h -> b h c l", l=self.block_len)
        B_blocks = rearrange(self.B, "b (c l) h n -> b c l h n", l=self.block_len)

        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)

        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        # Compute only the last-row decay weights: exp(prefix_last - prefix_c)
        # chunk_totals shape: [B, H, C+1] where first element is 0 (padded)
        chunk_totals = F.pad(A_cumsum[:, :, :, -1], (1, 0))
        chunk_prefix = torch.cumsum(chunk_totals, dim=-1)
        # final_decay: last prefix minus each prefix, shape [B, H, C+1]
        final_decay = torch.exp(chunk_prefix[:, :, -1:] - chunk_prefix)
        return torch.einsum("bhc,bchpn->bhpn", final_decay, states)
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
        return new_states[:, -1]
        # <<<END_IMPROVE>>>
