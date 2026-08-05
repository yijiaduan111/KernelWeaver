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
    return f'stark_cuda_l1_p97_{digest}'

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

torch::Tensor sdpa_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);

torch::Tensor sdpa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Expected 4D tensors");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Q, K, V must have same shape");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Tensors must be contiguous");
    return sdpa_forward_cuda(q, k, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sdpa_forward", &sdpa_forward, "Fused Scaled Dot-Product Attention");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Specialized kernel for L=512: one block per row, 512 threads
// Each thread handles exactly one column element
__global__ void scaled_softmax_512_kernel(
    const float* __restrict__ in_scores,
    float* __restrict__ out_scores,
    float scale,
    int L  // =512
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;  // 0..511

    const float* row_in  = in_scores  + (long)row * L;
    float*       row_out = out_scores + (long)row * L;

    // Each thread loads one element and scales it
    float val = row_in[tid] * scale;

    // Block-wide max reduction via warp reductions + shared memory
    // 512 threads / 32 = 16 warps
    __shared__ float smem[16];
    int lane = tid & 31;
    int warp = tid >> 5;

    float wmax = warp_reduce_max(val);
    if (lane == 0) smem[warp] = wmax;
    __syncthreads();

    // Final reduce across 16 warps using first warp (lanes 0..15)
    float row_max;
    if (warp == 0) {
        // Only lanes 0..15 are valid; lanes 16..31 get -inf
        float w = (lane < 16) ? smem[lane] : -1e38f;
        // Use full warp mask but only first 16 lanes have real data
        w = fmaxf(w, __shfl_down_sync(0xffffffff, w, 8));
        w = fmaxf(w, __shfl_down_sync(0xffffffff, w, 4));
        w = fmaxf(w, __shfl_down_sync(0xffffffff, w, 2));
        w = fmaxf(w, __shfl_down_sync(0xffffffff, w, 1));
        if (lane == 0) smem[0] = w;
    }
    __syncthreads();
    row_max = smem[0];

    // Compute exp
    float e = __expf(val - row_max);

    // Block-wide sum reduction
    float wsum = warp_reduce_sum(e);
    if (lane == 0) smem[warp] = wsum;
    __syncthreads();

    float row_sum;
    if (warp == 0) {
        float w = (lane < 16) ? smem[lane] : 0.0f;
        w += __shfl_down_sync(0xffffffff, w, 8);
        w += __shfl_down_sync(0xffffffff, w, 4);
        w += __shfl_down_sync(0xffffffff, w, 2);
        w += __shfl_down_sync(0xffffffff, w, 1);
        if (lane == 0) smem[0] = w;
    }
    __syncthreads();
    row_sum = smem[0];

    row_out[tid] = e / row_sum;
}

// Generic scaled softmax kernel for arbitrary L
__global__ void scaled_softmax_generic_kernel(
    float* __restrict__ scores,
    int L,
    float scale
) {
    int row = blockIdx.x;
    float* row_ptr = scores + (long)row * L;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float local_max = -1e38f;
    for (int j = tid; j < L; j += block_size) {
        float v = row_ptr[j] * scale;
        if (v > local_max) local_max = v;
    }

    __shared__ float smem[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (block_size + 31) / 32;

    local_max = warp_reduce_max(local_max);
    if (lane == 0) smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < num_warps) ? smem[lane] : -1e38f;
        v = warp_reduce_max(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int j = tid; j < L; j += block_size) {
        float v = __expf(row_ptr[j] * scale - row_max);
        row_ptr[j] = v;
        local_sum += v;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smem[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < num_warps) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    float row_sum = smem[0];

    float inv_sum = 1.0f / row_sum;
    for (int j = tid; j < L; j += block_size)
        row_ptr[j] *= inv_sum;
}

torch::Tensor sdpa_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    int B = q.size(0);
    int H = q.size(1);
    int L = q.size(2);
    int D = q.size(3);
    int BH = B * H;

    float scale = 1.0f / sqrtf((float)D);

    auto q3 = q.view({BH, L, D});
    auto k3 = k.view({BH, L, D});
    auto v3 = v.view({BH, L, D});

    // scores = Q @ K^T: [BH, L, L]
    auto scores = torch::bmm(q3, k3.transpose(1, 2));

    torch::Tensor attn;
    if (L == 512) {
        attn = torch::empty_like(scores);
        int blocks = BH * L;
        scaled_softmax_512_kernel<<<blocks, 512>>>(
            scores.data_ptr<float>(),
            attn.data_ptr<float>(),
            scale,
            L
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        // Generic path: in-place
        int threads = 256;
        int blocks = BH * L;
        scaled_softmax_generic_kernel<<<blocks, threads>>>(
            scores.data_ptr<float>(),
            L,
            scale
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        attn = scores;
    }

    auto out3 = torch::bmm(attn, v3);
    return out3.view({B, H, L, D});
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if (
            Q.is_cuda and K.is_cuda and V.is_cuda and
            Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32 and
            Q.dim() == 4 and K.dim() == 4 and V.dim() == 4 and
            Q.shape == K.shape and Q.shape == V.shape and
            Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        ):
            out = _stark_get_extension().sdpa_forward(Q, K, V)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return out
        # <<<END_IMPROVE>>>
