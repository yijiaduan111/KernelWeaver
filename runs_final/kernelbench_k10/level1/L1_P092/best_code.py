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
    return f'stark_cuda_l1_p92_{digest}'

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
#include <ATen/cuda/CUDAContext.h>

torch::Tensor exclusive_cumsum_cuda(torch::Tensor x);

torch::Tensor exclusive_cumsum(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "Input must have at least 1 dimension");

    int64_t ndim = x.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension");
    TORCH_CHECK(dim == ndim - 1, "Only last dimension supported");

    return exclusive_cumsum_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exclusive_cumsum", &exclusive_cumsum, "exclusive cumsum fast path");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

__device__ __forceinline__ float warp_scan_inclusive(float val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += temp;
        }
    }
    return val;
}

__global__ void exclusive_cumsum_kernel_scalar(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int64_t rows,
                                               int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float carry = 0.0f;

    for (int64_t tile_start = 0; tile_start < cols; tile_start += BLOCK_SIZE) {
        int tid = threadIdx.x;
        int64_t idx = tile_start + tid;

        float val = (idx < cols) ? row_input[idx] : 0.0f;

        float warp_val = warp_scan_inclusive(val);

        int warp_id = tid / WARP_SIZE;
        int lane_id = tid & (WARP_SIZE - 1);

        if (lane_id == WARP_SIZE - 1) {
            warp_sums[warp_id] = warp_val;
        }
        __syncthreads();

        if (tid < BLOCK_SIZE / WARP_SIZE) {
            float warp_sum = warp_sums[tid];
            warp_sum = warp_scan_inclusive(warp_sum);
            warp_sums[tid] = warp_sum;
        }
        __syncthreads();

        float block_prefix = (warp_id > 0) ? warp_sums[warp_id - 1] : 0.0f;
        float inclusive_val = warp_val + block_prefix;

        float exclusive_val = inclusive_val - val + carry;

        if (idx < cols) {
            row_output[idx] = exclusive_val;
        }

        float tile_sum = warp_sums[(BLOCK_SIZE / WARP_SIZE) - 1];
        carry += tile_sum;

        __syncthreads();
    }
}

__global__ void exclusive_cumsum_kernel_float4(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int64_t rows,
                                               int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    const float4* row_input = reinterpret_cast<const float4*>(input + row * cols);
    float4* row_output = reinterpret_cast<float4*>(output + row * cols);
    int64_t vec_cols = cols / 4;

    float carry = 0.0f;

    for (int64_t tile_start = 0; tile_start < vec_cols; tile_start += BLOCK_SIZE) {
        int tid = threadIdx.x;
        int64_t idx = tile_start + tid;

        float4 vec_val = {0.0f, 0.0f, 0.0f, 0.0f};
        if (idx < vec_cols) {
            vec_val = row_input[idx];
        }

        float local_sum0 = vec_val.x;
        float local_sum1 = local_sum0 + vec_val.y;
        float local_sum2 = local_sum1 + vec_val.z;
        float local_sum3 = local_sum2 + vec_val.w;

        float thread_total = local_sum3;
        float warp_val = warp_scan_inclusive(thread_total);

        int warp_id = tid / WARP_SIZE;
        int lane_id = tid & (WARP_SIZE - 1);

        if (lane_id == WARP_SIZE - 1) {
            warp_sums[warp_id] = warp_val;
        }
        __syncthreads();

        if (tid < BLOCK_SIZE / WARP_SIZE) {
            float warp_sum = warp_sums[tid];
            warp_sum = warp_scan_inclusive(warp_sum);
            warp_sums[tid] = warp_sum;
        }
        __syncthreads();

        float block_prefix = (warp_id > 0) ? warp_sums[warp_id - 1] : 0.0f;
        float thread_prefix = warp_val - thread_total + block_prefix + carry;

        float4 result;
        result.x = thread_prefix;
        result.y = thread_prefix + local_sum0;
        result.z = thread_prefix + local_sum1;
        result.w = thread_prefix + local_sum2;

        if (idx < vec_cols) {
            row_output[idx] = result;
        }

        float tile_sum = warp_sums[(BLOCK_SIZE / WARP_SIZE) - 1];
        carry += tile_sum;

        __syncthreads();
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor x) {
    int64_t cols = x.size(-1);
    if (cols == 0) {
        return torch::empty_like(x);
    }

    int64_t rows = x.numel() / cols;

    auto output = torch::empty_like(x);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    bool can_vectorize = (cols % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(output_ptr) % 16 == 0);

    if (can_vectorize) {
        exclusive_cumsum_kernel_float4<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input_ptr, output_ptr, rows, cols
        );
    } else {
        exclusive_cumsum_kernel_scalar<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input_ptr, output_ptr, rows, cols
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs an exclusive cumulative sum (does not include the current element).

        Parameters:
            dim (int): The dimension along which to perform the exclusive cumulative sum.
        """
    def __init__(self, dim):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.dim = dim
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        dim = self.dim if self.dim >= 0 else self.dim + x.dim()
        if (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and
            x.dim() >= 1 and 0 <= dim < x.dim() and dim == x.dim() - 1):
            return _stark_get_extension().exclusive_cumsum(x, dim)
        cumsum = torch.cumsum(x.narrow(dim=dim, start=0, length=x.size(dim)-1), dim=dim)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), cumsum), dim=self.dim)
        # <<<END_IMPROVE>>>
