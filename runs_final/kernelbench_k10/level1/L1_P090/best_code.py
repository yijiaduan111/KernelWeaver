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
    return f'stark_cuda_l1_p90_{digest}'

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

torch::Tensor cumprod_lastdim_vec4_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumprod_lastdim_vec4_cuda", &cumprod_lastdim_vec4_cuda, "Vectorized last-dim cumprod (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define NWARPS (BLOCK_SIZE / 32)

// Specialized kernel: assumes ncols is an exact multiple of BLOCK_SIZE*4.
// No bounds checks, no scalar tail, simplified carry broadcast.
__global__ void cumprod_vec4_aligned_kernel(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             int chunks) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    __shared__ float warp_last[NWARPS];
    __shared__ float warp_prefix[NWARPS];
    __shared__ float bc_shared;

    const float* row_in  = in  + (long long)row * (chunks * 4);
    float*       row_out = out + (long long)row * (chunks * 4);

    float block_carry = 1.0f;

    for (int base = 0; base < chunks; base += BLOCK_SIZE) {
        int chunk_idx = base + tid;
        const float4 vec = reinterpret_cast<const float4*>(row_in)[chunk_idx];
        float v0 = vec.x;
        float v1 = vec.y * v0;
        float v2 = vec.z * v1;
        float v3 = vec.w * v2;

        // inclusive warp scan on v3
        float val = v3;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float n = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) val *= n;
        }
        float excl = __shfl_up_sync(0xffffffff, val, 1);
        if (lane == 0) excl = 1.0f;

        if (lane == 31) warp_last[wid] = val;
        __syncthreads();

        // inter-warp exclusive prefix in warp 0
        if (wid == 0) {
            float w = (lane < NWARPS) ? warp_last[lane] : 1.0f;
            float ws = w;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float n2 = __shfl_up_sync(0xffffffff, ws, offset);
                if (lane >= offset) ws *= n2;
            }
            float we = __shfl_up_sync(0xffffffff, ws, 1);
            if (lane == 0) we = 1.0f;
            if (lane < NWARPS) warp_prefix[lane] = we;
            // store total for block_carry update
            if (lane == NWARPS - 1) bc_shared = ws;  // inclusive of last warp
        }
        __syncthreads();

        float carry_in = block_carry * warp_prefix[wid] * excl;
        float4 res;
        res.x = carry_in * v0;
        res.y = carry_in * v1;
        res.z = carry_in * v2;
        res.w = carry_in * v3;
        reinterpret_cast<float4*>(row_out)[chunk_idx] = res;

        // update block_carry: last thread's warp_prefix[last_warp] * warp_last[last_warp]
        block_carry = block_carry * bc_shared;
        __syncthreads();
    }
}

// Generic kernel: handles any ncols (with tail and partial tiles)
__global__ void cumprod_vec4_generic_kernel(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             int ncols) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    __shared__ float warp_last[NWARPS];
    __shared__ float warp_prefix[NWARPS];

    const float* row_in  = in  + (long long)row * ncols;
    float*       row_out = out + (long long)row * ncols;

    float block_carry = 1.0f;
    int chunks = ncols / 4;

    for (int base = 0; base < chunks; base += BLOCK_SIZE) {
        int chunk_idx = base + tid;
        float v0, v1, v2, v3;
        if (chunk_idx < chunks) {
            const float4 vec = reinterpret_cast<const float4*>(row_in)[chunk_idx];
            v0 = vec.x; v1 = vec.y * v0; v2 = vec.z * v1; v3 = vec.w * v2;
        } else {
            v0 = v1 = v2 = v3 = 1.0f;
        }

        float val = v3;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float n = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) val *= n;
        }
        float excl = __shfl_up_sync(0xffffffff, val, 1);
        if (lane == 0) excl = 1.0f;

        if (lane == 31) warp_last[wid] = val;
        __syncthreads();

        if (wid == 0) {
            float w = (lane < NWARPS) ? warp_last[lane] : 1.0f;
            float ws = w;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float n2 = __shfl_up_sync(0xffffffff, ws, offset);
                if (lane >= offset) ws *= n2;
            }
            float we = __shfl_up_sync(0xffffffff, ws, 1);
            if (lane == 0) we = 1.0f;
            if (lane < NWARPS) warp_prefix[lane] = we;
        }
        __syncthreads();

        float carry_in = block_carry * warp_prefix[wid] * excl;

        if (chunk_idx < chunks) {
            float4 res;
            res.x = carry_in * v0;
            res.y = carry_in * v1;
            res.z = carry_in * v2;
            res.w = carry_in * v3;
            reinterpret_cast<float4*>(row_out)[chunk_idx] = res;
        }

        int last_active = min(BLOCK_SIZE, chunks - base) - 1;
        __shared__ float bc_shared;
        if (tid == last_active) bc_shared = block_carry * warp_prefix[wid] * val;
        __syncthreads();
        block_carry = bc_shared;
    }

    // scalar tail
    if (tid == 0) {
        float carry = block_carry;
        for (int i = chunks * 4; i < ncols; ++i) {
            carry *= row_in[i];
            row_out[i] = carry;
        }
    }
}

torch::Tensor cumprod_lastdim_vec4_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto out = torch::empty_like(x);
    const int nrows = (int)x.size(0);
    const int ncols = (int)x.size(1);

    if (nrows == 0 || ncols == 0) return out;

    dim3 grid(nrows);
    dim3 block(BLOCK_SIZE);

    if (ncols % (BLOCK_SIZE * 4) == 0) {
        int chunks = ncols / 4;
        cumprod_vec4_aligned_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            chunks
        );
    } else {
        cumprod_vec4_generic_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            ncols
        );
    }
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a cumulative product operation along a specified dimension.

        Parameters:
            dim (int): The dimension along which to perform the cumulative product operation.
        """
    def __init__(self, dim):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initialize the CumulativeProductModel.

                Args:
                    dim (int): The dimension along which to perform the cumulative product.
                """
        self.dim = dim
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass, computing the cumulative product along the specified dimension.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

                Returns:
                    torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and self.dim == x.dim() - 1 and x.is_contiguous()):
            return _stark_get_extension().cumprod_lastdim_vec4_cuda(x)
        return torch.cumprod(x, dim=self.dim)
        # <<<END_IMPROVE>>>
