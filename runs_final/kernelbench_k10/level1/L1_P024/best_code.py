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
    return f'stark_cuda_l1_p24_{digest}'

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

torch::Tensor logsoftmax_forward_cuda(torch::Tensor x);

torch::Tensor logsoftmax_forward(torch::Tensor x) {
    return logsoftmax_forward_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("logsoftmax_forward", &logsoftmax_forward, "logsoftmax_forward");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Combine two (max, sum) pairs in a numerically stable way
__device__ __forceinline__ void combine_max_sum(float a_max, float a_sum, float b_max, float b_sum,
                                                  float& out_max, float& out_sum) {
    if (a_max >= b_max) {
        out_max = a_max;
        out_sum = a_sum + b_sum * __expf(b_max - a_max);
    } else {
        out_max = b_max;
        out_sum = b_sum + a_sum * __expf(a_max - b_max);
    }
}

// Warp-level reduction for (max, sum) pairs
__device__ __forceinline__ void warp_reduce_max_sum(float& val_max, float& val_sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, val_max, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, val_sum, offset);
        combine_max_sum(val_max, val_sum, other_max, other_sum, val_max, val_sum);
    }
}

// One block per row, 512 threads, float4 vectorized loads
// Two float4 per iteration in hot loops to increase memory-level parallelism
__global__ void __launch_bounds__(512, 2) logsoftmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;  // 512
    const int nwarps = nthreads / 32; // 16
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem_max[16];
    __shared__ float smem_sum[16];

    const float* row_in = input + (long long)row * cols;
    float* row_out = output + (long long)row * cols;

    int cols4 = cols / 4;
    // cols8: largest even multiple of nthreads we can cover with 2x float4 per iter
    int cols8 = (cols4 / 2) * 2;  // round down to even
    // Align cols8 so that i and i+nthreads are both < cols8 in the dual loop
    // We iterate i in [tid, cols8, 2*nthreads), accessing i and i+nthreads
    // Need i+nthreads < cols4, so cols8 = cols4 & ~1 is not quite right.
    // Correct: the dual loop covers indices [0, cols4) in steps of 2*nthreads per thread.
    // Each thread handles i and i+nthreads where i = tid + k*2*nthreads.
    // Both i and i+nthreads must be < cols4.
    // So the dual loop runs while i + nthreads < cols4, i.e. i < cols4 - nthreads.
    const float4* row_in4 = reinterpret_cast<const float4*>(row_in);

    // Online pass: compute per-thread (max, sum)
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    // Dual float4 loop: each iteration processes two float4 chunks
    int i = tid;
    for (; i + nthreads < cols4; i += 2 * nthreads) {
        float4 v0 = __ldg(row_in4 + i);
        float4 v1 = __ldg(row_in4 + i + nthreads);

        // Update with v0
        float lmax0 = fmaxf(fmaxf(v0.x, v0.y), fmaxf(v0.z, v0.w));
        if (lmax0 >= thread_max) {
            thread_sum = thread_sum * __expf(thread_max - lmax0)
                       + __expf(v0.x - lmax0) + __expf(v0.y - lmax0)
                       + __expf(v0.z - lmax0) + __expf(v0.w - lmax0);
            thread_max = lmax0;
        } else {
            thread_sum += __expf(v0.x - thread_max) + __expf(v0.y - thread_max)
                        + __expf(v0.z - thread_max) + __expf(v0.w - thread_max);
        }

        // Update with v1
        float lmax1 = fmaxf(fmaxf(v1.x, v1.y), fmaxf(v1.z, v1.w));
        if (lmax1 >= thread_max) {
            thread_sum = thread_sum * __expf(thread_max - lmax1)
                       + __expf(v1.x - lmax1) + __expf(v1.y - lmax1)
                       + __expf(v1.z - lmax1) + __expf(v1.w - lmax1);
            thread_max = lmax1;
        } else {
            thread_sum += __expf(v1.x - thread_max) + __expf(v1.y - thread_max)
                        + __expf(v1.z - thread_max) + __expf(v1.w - thread_max);
        }
    }
    // Cleanup single float4 loop for remaining chunks
    for (; i < cols4; i += nthreads) {
        float4 v = __ldg(row_in4 + i);
        float lmax = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        if (lmax >= thread_max) {
            thread_sum = thread_sum * __expf(thread_max - lmax)
                       + __expf(v.x - lmax) + __expf(v.y - lmax)
                       + __expf(v.z - lmax) + __expf(v.w - lmax);
            thread_max = lmax;
        } else {
            thread_sum += __expf(v.x - thread_max) + __expf(v.y - thread_max)
                        + __expf(v.z - thread_max) + __expf(v.w - thread_max);
        }
    }
    // Handle scalar tail
    for (int j = cols4 * 4 + tid; j < cols; j += nthreads) {
        float val = __ldg(row_in + j);
        combine_max_sum(thread_max, thread_sum, val, 1.0f, thread_max, thread_sum);
    }

    // Warp reduce (max, sum)
    warp_reduce_max_sum(thread_max, thread_sum);
    if (lane_id == 0) {
        smem_max[warp_id] = thread_max;
        smem_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Block reduce (max, sum) using first warp
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    if (tid < nwarps) {
        row_max = smem_max[tid];
        row_sum = smem_sum[tid];
    }
    if (warp_id == 0) {
        warp_reduce_max_sum(row_max, row_sum);
    }
    if (tid == 0) {
        smem_max[0] = row_max;
        smem_sum[0] = row_sum;
    }
    __syncthreads();
    row_max = smem_max[0];
    row_sum = smem_sum[0];

    float log_denom = row_max + __logf(row_sum);

    // Write pass: dual float4 loop
    float4* row_out4 = reinterpret_cast<float4*>(row_out);
    int wi = tid;
    for (; wi + nthreads < cols4; wi += 2 * nthreads) {
        float4 v0 = __ldg(row_in4 + wi);
        float4 v1 = __ldg(row_in4 + wi + nthreads);
        float4 o0, o1;
        o0.x = v0.x - log_denom; o0.y = v0.y - log_denom;
        o0.z = v0.z - log_denom; o0.w = v0.w - log_denom;
        o1.x = v1.x - log_denom; o1.y = v1.y - log_denom;
        o1.z = v1.z - log_denom; o1.w = v1.w - log_denom;
        row_out4[wi] = o0;
        row_out4[wi + nthreads] = o1;
    }
    // Cleanup single float4 write loop
    for (; wi < cols4; wi += nthreads) {
        float4 v = __ldg(row_in4 + wi);
        float4 o;
        o.x = v.x - log_denom; o.y = v.y - log_denom;
        o.z = v.z - log_denom; o.w = v.w - log_denom;
        row_out4[wi] = o;
    }
    // Scalar tail
    for (int j = cols4 * 4 + tid; j < cols; j += nthreads) {
        row_out[j] = __ldg(row_in + j) - log_denom;
    }
}

torch::Tensor logsoftmax_forward_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");

    int rows = x.size(0);
    int cols = x.size(1);
    auto output = torch::empty_like(x);

    dim3 grid(rows);
    dim3 block(512);
    logsoftmax_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        cols
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a LogSoftmax activation.
        """
    def __init__(self, dim: int = 1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.dim = dim
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies LogSoftmax activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, dim).

                Returns:
                    torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        dim_normalized = self.dim if self.dim >= 0 else self.dim + x.dim()
        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 2
                and x.is_contiguous() and dim_normalized == x.dim() - 1):
            return _stark_get_extension().logsoftmax_forward(x)
        return torch.log_softmax(x, dim=self.dim)
        # <<<END_IMPROVE>>>
