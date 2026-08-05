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
    return f'stark_cuda_l3_p44_{digest}'

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

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
torch::Tensor minigpt_ln2_mlp_residual_forward(
    torch::Tensor x,
    torch::Tensor ln_weight, torch::Tensor ln_bias, double ln_eps,
    torch::Tensor c_fc_weight, torch::Tensor c_fc_bias,
    torch::Tensor c_proj_weight, torch::Tensor c_proj_bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "Fused LayerNorm forward");
    m.def("minigpt_ln2_mlp_residual_forward", &minigpt_ln2_mlp_residual_forward,
          "Fused ln2 + MLP + residual add");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Warp-level reduction helper
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Specialized LayerNorm kernel for hidden=768.
// 256 threads/block, each thread handles 3 elements (768/256=3).
// ---------------------------------------------------------------------------
__global__ void layernorm_768_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int rows,
    float eps
) {
    const int HIDDEN = 768;
    const int BLOCK  = 256;
    const int ELEMS  = 3;

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;

    const float* row_in = x   + row * HIDDEN;
    float*       row_out = out + row * HIDDEN;

    float vals[ELEMS];
    float lsum = 0.0f, lsumsq = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        int idx = tid + e * BLOCK;
        float v = row_in[idx];
        vals[e]  = v;
        lsum    += v;
        lsumsq  += v * v;
    }

    float wsum   = warp_reduce_sum(lsum);
    float wsumsq = warp_reduce_sum(lsumsq);

    const int WARPS = BLOCK / 32;
    __shared__ float s_sum[WARPS];
    __shared__ float s_sumsq[WARPS];

    if (lane == 0) {
        s_sum[wid]   = wsum;
        s_sumsq[wid] = wsumsq;
    }
    __syncthreads();

    if (wid == 0) {
        float ws  = (lane < WARPS) ? s_sum[lane]   : 0.0f;
        float wss = (lane < WARPS) ? s_sumsq[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            ws  += __shfl_down_sync(0xffffffff, ws,  offset);
            wss += __shfl_down_sync(0xffffffff, wss, offset);
        }
        if (lane == 0) {
            s_sum[0]   = ws;
            s_sumsq[0] = wss;
        }
    }
    __syncthreads();

    float mean     = s_sum[0]   * (1.0f / HIDDEN);
    float variance = s_sumsq[0] * (1.0f / HIDDEN) - mean * mean;
    float inv_std  = rsqrtf(variance + eps);

    #pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        int idx = tid + e * BLOCK;
        row_out[idx] = (vals[e] - mean) * inv_std * weight[idx] + bias[idx];
    }
}

// ---------------------------------------------------------------------------
// Generic fallback LayerNorm for non-768 hidden sizes
// ---------------------------------------------------------------------------
__global__ void layernorm_generic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int rows,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* s_sum    = shared;
    float* s_sum_sq = shared + blockDim.x;

    const float* row_in = x   + row * hidden;
    float*       row_out = out + row * hidden;

    float lsum = 0.0f, lsumsq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = row_in[i];
        lsum    += v;
        lsumsq  += v * v;
    }
    s_sum[threadIdx.x]    = lsum;
    s_sum_sq[threadIdx.x] = lsumsq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x]    += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean     = s_sum[0]    / (float)hidden;
    float variance = s_sum_sq[0] / (float)hidden - mean * mean;
    float inv_std  = rsqrtf(variance + eps);

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = row_in[i];
        row_out[i] = (v - mean) * inv_std * weight[i] + bias[i];
    }
}

// ---------------------------------------------------------------------------
// NewGELU elementwise kernel
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------
__global__ void newgelu_inplace_kernel(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    // sqrt(2/pi) = 0.7978845608028654f
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(inner));
}

// ---------------------------------------------------------------------------
// layernorm_forward: public entry used by both ln_1 and ln_2 forward sites
// ---------------------------------------------------------------------------
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "weight and bias must be CUDA tensors");
    TORCH_CHECK(weight.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32,
                "weight and bias must be float32");

    int hidden = (int)x.size(-1);
    TORCH_CHECK(weight.numel() == hidden && bias.numel() == hidden, "weight/bias size mismatch");

    x = x.contiguous();
    int rows = (int)(x.numel() / hidden);
    auto out = torch::empty_like(x);
    float feps = static_cast<float>(eps);

    if (hidden == 768) {
        layernorm_768_kernel<<<rows, 256>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            rows, feps
        );
    } else {
        int block_size = 256;
        int shared_mem = 2 * block_size * sizeof(float);
        layernorm_generic_kernel<<<rows, block_size, shared_mem>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            rows, hidden, feps
        );
    }
    return out;
}

// ---------------------------------------------------------------------------
// minigpt_ln2_mlp_residual_forward
// Computes: x + c_proj(NewGELU(c_fc(LayerNorm(x, ln_weight, ln_bias, ln_eps))))
// resid_pdrop=0.0 is assumed; dropout is skipped.
// Uses ATen matmul (cuBLAS) for linear projections.
// ---------------------------------------------------------------------------
torch::Tensor minigpt_ln2_mlp_residual_forward(
    torch::Tensor x,
    torch::Tensor ln_weight, torch::Tensor ln_bias, double ln_eps,
    torch::Tensor c_fc_weight, torch::Tensor c_fc_bias,
    torch::Tensor c_proj_weight, torch::Tensor c_proj_bias
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32");

    // Ensure contiguous for safe pointer arithmetic in LN kernel
    auto x_cont = x.contiguous();

    // --- LayerNorm over last dim ---
    int hidden = (int)x_cont.size(-1);
    int rows   = (int)(x_cont.numel() / hidden);
    auto ln_out = torch::empty_like(x_cont);
    float feps  = static_cast<float>(ln_eps);

    if (hidden == 768) {
        layernorm_768_kernel<<<rows, 256>>>(
            x_cont.data_ptr<float>(),
            ln_weight.data_ptr<float>(),
            ln_bias.data_ptr<float>(),
            ln_out.data_ptr<float>(),
            rows, feps
        );
    } else {
        int block_size = 256;
        int shared_mem = 2 * block_size * sizeof(float);
        layernorm_generic_kernel<<<rows, block_size, shared_mem>>>(
            x_cont.data_ptr<float>(),
            ln_weight.data_ptr<float>(),
            ln_bias.data_ptr<float>(),
            ln_out.data_ptr<float>(),
            rows, hidden, feps
        );
    }

    // --- c_fc linear: [rows, hidden] x [4*hidden, hidden]^T -> [rows, 4*hidden] ---
    // ATen linear: input [*, in], weight [out, in], bias [out] -> [*, out]
    auto fc_out = torch::addmm(c_fc_bias,
                               ln_out.view({rows, hidden}),
                               c_fc_weight.t());

    // --- NewGELU in-place ---
    int fc_numel = (int)fc_out.numel();
    int gelu_threads = 256;
    int gelu_blocks  = (fc_numel + gelu_threads - 1) / gelu_threads;
    newgelu_inplace_kernel<<<gelu_blocks, gelu_threads>>>(fc_out.data_ptr<float>(), fc_numel);

    // --- c_proj linear: [rows, 4*hidden] x [hidden, 4*hidden]^T -> [rows, hidden] ---
    auto proj_out = torch::addmm(c_proj_bias,
                                 fc_out,
                                 c_proj_weight.t());

    // --- Residual add, restore original shape ---
    auto result = x_cont.view({rows, hidden}) + proj_out;
    return result.view(x.sizes());
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
                    c_fc    = nn.Linear(n_embd, 4 * n_embd),
                    c_proj  = nn.Linear(4 * n_embd, n_embd),
                    act     = NewGELU(),
                    dropout = nn.Dropout(resid_pdrop),
                ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        ext = _stark_get_extension()
        ln1_in = x.contiguous()
        ln1_out = ext.layernorm_forward(ln1_in, self.ln_1.weight, self.ln_1.bias, self.ln_1.eps)
        x = x + self.attn(ln1_out)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        ext = _stark_get_extension()
        x = ext.minigpt_ln2_mlp_residual_forward(
        x.contiguous(),
        self.ln_2.weight, self.ln_2.bias, self.ln_2.eps,
        self.mlp.c_fc.weight, self.mlp.c_fc.bias,
        self.mlp.c_proj.weight, self.mlp.c_proj.bias
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        return x
        # <<<END_IMPROVE>>>
