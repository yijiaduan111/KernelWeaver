import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    return f'stark_cuda_l3_p50_{digest}'

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

torch::Tensor scale_mask_relu_inplace(
    torch::Tensor att,
    int64_t T,
    double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_mask_relu_inplace", &scale_mask_relu_inplace,
          "Fused in-place scale + causal mask + ReLU for attention (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void scale_mask_relu_inplace_kernel(
    float4* __restrict__ data,
    int T,
    float scale,
    int num_vec_cols
) {
    // blockIdx.z = batch_head index
    // blockIdx.y = attention row index (0..T-1)
    // blockIdx.x * blockDim.x + threadIdx.x = vec_col index
    int row = blockIdx.y;
    int batch_head = blockIdx.z;
    int vec_col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_col_idx >= num_vec_cols) return;

    int offset = (batch_head * T + row) * num_vec_cols + vec_col_idx;
    int diag_vec = row >> 2;  // row / 4

    if (vec_col_idx < diag_vec) {
        // Fully below diagonal: all 4 elements are valid (col <= row)
        float4 vals = data[offset];
        float x = vals.x * scale;
        float y = vals.y * scale;
        float z = vals.z * scale;
        float w = vals.w * scale;
        vals.x = x < 0.0f ? 0.0f : x;
        vals.y = y < 0.0f ? 0.0f : y;
        vals.z = z < 0.0f ? 0.0f : z;
        vals.w = w < 0.0f ? 0.0f : w;
        data[offset] = vals;
    } else if (vec_col_idx == diag_vec) {
        // Diagonal vector: per-lane causal masking
        int base_col = vec_col_idx * 4;
        float4 vals = data[offset];
        float x = vals.x * scale;
        float y = vals.y * scale;
        float z = vals.z * scale;
        float w = vals.w * scale;
        vals.x = (base_col + 0 > row || x < 0.0f) ? 0.0f : x;
        vals.y = (base_col + 1 > row || y < 0.0f) ? 0.0f : y;
        vals.z = (base_col + 2 > row || z < 0.0f) ? 0.0f : z;
        vals.w = (base_col + 3 > row || w < 0.0f) ? 0.0f : w;
        data[offset] = vals;
    } else {
        // Fully above diagonal: zero out without reading
        data[offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

torch::Tensor scale_mask_relu_inplace(
    torch::Tensor att,
    int64_t T,
    double scale
) {
    TORCH_CHECK(att.is_cuda(), "att must be a CUDA tensor");
    TORCH_CHECK(att.is_contiguous(), "att must be contiguous");
    TORCH_CHECK(att.scalar_type() == torch::kFloat32, "att must be float32");
    TORCH_CHECK(T % 4 == 0, "T must be divisible by 4 for float4 vectorization");

    int num_vec_cols = static_cast<int>(T / 4);
    int total_rows = static_cast<int>(att.numel() / T);
    int batch_head_count = total_rows / static_cast<int>(T);

    const int threads = 256;
    int grid_x = (num_vec_cols + threads - 1) / threads;
    // gridDim.y = T (one block-row per attention row)
    // gridDim.z = batch_head_count (B * n_head)
    dim3 blocks(grid_x, static_cast<int>(T), batch_head_count);

    scale_mask_relu_inplace_kernel<<<blocks, threads>>>(
        reinterpret_cast<float4*>(att.data_ptr<float>()),
        static_cast<int>(T),
        static_cast<float>(scale),
        num_vec_cols
    );

    return att;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
        It is possible to use torch.nn.MultiheadAttention here but I am including an
        explicit implementation here to show that there is nothing too scary here.
        """
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                             .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        B, T, C = x.size()
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        att = q @ k.transpose(-2, -1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        att = _stark_get_extension().scale_mask_relu_inplace(att.contiguous(), T, 1.0 / math.sqrt(k.size(-1)))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        # scale, causal mask, and ReLU already applied in the fused CUDA kernel above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        y = att @ v
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_10>>>
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_11>>>
        return y
        # <<<END_IMPROVE>>>
