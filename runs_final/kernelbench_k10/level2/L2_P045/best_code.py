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
    return f'stark_cuda_l2_p45_{digest}'

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

torch::Tensor row_logsumexp_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_logsumexp", &row_logsumexp_cuda, "Row-wise logsumexp over dim=1 (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Generic kernel for arbitrary column counts, 256 threads per block.
__global__ void row_logsumexp_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * cols;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    // Pass 1: thread-local max
    float tmax = -1e38f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = row_ptr[c];
        if (v > tmax) tmax = v;
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        tmax = fmaxf(tmax, __shfl_down_sync(0xffffffff, tmax, offset));

    __shared__ float smem[32];
    if (lane == 0) smem[warp] = tmax;
    __syncthreads();

    // Block reduce max across 8 warps (256 threads)
    if (tid < 8) {
        tmax = smem[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            tmax = fmaxf(tmax, __shfl_down_sync(0x000000ff, tmax, offset));
    }
    if (tid == 0) smem[0] = tmax;
    __syncthreads();
    float row_max = smem[0];

    // Pass 2: thread-local sum of exp(x - max)
    float tsum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x)
        tsum += __expf(row_ptr[c] - row_max);

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        tsum += __shfl_down_sync(0xffffffff, tsum, offset);

    if (lane == 0) smem[warp] = tsum;
    __syncthreads();

    if (tid < 8) {
        tsum = smem[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            tsum += __shfl_down_sync(0x000000ff, tsum, offset);
    }

    if (tid == 0)
        out[row] = row_max + __logf(tsum);
}

// Specialized kernel for cols == 1024: 256 threads, each loads exactly 4 scalars.
// Thread tid handles indices: tid, tid+256, tid+512, tid+768.
__global__ void row_logsumexp_1024_fixed4_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * 1024;
    int tid = threadIdx.x;  // 0..255
    int lane = tid & 31;
    int warp = tid >> 5;    // 0..7

    // Each thread loads 4 scalars at stride 256
    float v0 = row_ptr[tid];
    float v1 = row_ptr[tid + 256];
    float v2 = row_ptr[tid + 512];
    float v3 = row_ptr[tid + 768];

    // Thread-local max over 4 values
    float tmax = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));

    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        tmax = fmaxf(tmax, __shfl_down_sync(0xffffffff, tmax, offset));

    __shared__ float smem[8];
    if (lane == 0) smem[warp] = tmax;
    __syncthreads();

    // Block reduce max across 8 warps â done by first 8 threads (one per warp)
    if (tid < 8) {
        tmax = smem[tid];
        // 8-way reduce: offset 4, 2, 1
        tmax = fmaxf(tmax, __shfl_down_sync(0x000000ff, tmax, 4));
        tmax = fmaxf(tmax, __shfl_down_sync(0x000000ff, tmax, 2));
        tmax = fmaxf(tmax, __shfl_down_sync(0x000000ff, tmax, 1));
    }
    if (tid == 0) smem[0] = tmax;
    __syncthreads();
    float row_max = smem[0];

    // Thread-local sum of exp(v - row_max) over the same 4 values
    float tsum = __expf(v0 - row_max)
               + __expf(v1 - row_max)
               + __expf(v2 - row_max)
               + __expf(v3 - row_max);

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        tsum += __shfl_down_sync(0xffffffff, tsum, offset);

    if (lane == 0) smem[warp] = tsum;
    __syncthreads();

    if (tid < 8) {
        tsum = smem[tid];
        tsum += __shfl_down_sync(0x000000ff, tsum, 4);
        tsum += __shfl_down_sync(0x000000ff, tsum, 2);
        tsum += __shfl_down_sync(0x000000ff, tsum, 1);
    }

    if (tid == 0)
        out[row] = row_max + __logf(tsum);
}

torch::Tensor row_logsumexp_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int rows = (int)x.size(0);
    int cols = (int)x.size(1);
    auto out = torch::empty({rows}, x.options());

    if (cols == 1024) {
        row_logsumexp_1024_fixed4_kernel<<<rows, 256>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            rows
        );
    } else {
        row_logsumexp_kernel<<<rows, 256>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            rows, cols
        );
    }
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a matrix multiplication (Gemm), applies Sigmoid,
        another Gemm, and computes LogSumExp over features.
        """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.linear1(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = torch.sigmoid(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.linear2(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous():
            x = _stark_get_extension().row_logsumexp(x)
        else:
            x = torch.logsumexp(x, dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
