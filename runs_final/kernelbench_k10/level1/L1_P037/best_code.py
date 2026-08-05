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
    return f'stark_cuda_l1_p37_{digest}'

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

torch::Tensor frobenius_normalize_cuda(torch::Tensor x);

torch::Tensor frobenius_normalize(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    return frobenius_normalize_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("frobenius_normalize", &frobenius_normalize, "Frobenius normalize (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel 1: compute partial sum of squares, one partial per block.
__global__ void sum_sq_kernel(const float* __restrict__ x, double* __restrict__ partials, int64_t N) {
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    double acc = 0.0;
    for (int64_t i = idx; i < N; i += stride) {
        double v = (double)x[i];
        acc += v * v;
    }
    smem[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = smem[0];
}

// Kernel 2: reduce partial sums into a single scalar.
__global__ void reduce_partials_kernel(const double* __restrict__ partials, double* __restrict__ result, int n_partials) {
    __shared__ double smem[1024];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int i = tid; i < n_partials; i += blockDim.x) {
        acc += partials[i];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) result[0] = smem[0];
}

// Kernel 3: normalize in place using dual float4 vectorized loads/stores per iteration.
// Each thread processes 2 float4 values (8 floats) per loop iteration to increase MLP.
__global__ void normalize_kernel_f4_inplace(float* __restrict__ x, int64_t N4, int64_t N, const double* __restrict__ sum_sq_ptr) {
    float inv_norm = (float)(1.0 / sqrt(*sum_sq_ptr));

    float4* x4 = reinterpret_cast<float4*>(x);

    // Dual-float4 unrolled loop: each thread handles 2 float4 per iteration
    int64_t tid_global = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t vec_stride = (int64_t)gridDim.x * blockDim.x * 2;
    int64_t base = tid_global * 2;

    for (int64_t i = base; i + 1 < N4; i += vec_stride) {
        float4 v0 = x4[i];
        float4 v1 = x4[i + 1];
        v0.x *= inv_norm; v0.y *= inv_norm; v0.z *= inv_norm; v0.w *= inv_norm;
        v1.x *= inv_norm; v1.y *= inv_norm; v1.z *= inv_norm; v1.w *= inv_norm;
        x4[i]     = v0;
        x4[i + 1] = v1;
    }

    // Cleanup: handle one remaining float4 if N4 is odd relative to this thread's coverage
    // After the dual loop, check if base is still < N4 (handles the single leftover)
    // We need a single-step cleanup pass over all remaining float4 chunks.
    // Use a separate grid-stride pass with stride = gridDim.x * blockDim.x
    int64_t single_stride = (int64_t)gridDim.x * blockDim.x;
    // Find the start of the cleanup region: largest even multiple of single_stride * 2 covered
    // Simpler: just do a full single-stride pass but skip already-processed indices.
    // Instead, compute the boundary: all indices i < N4 not covered by the dual loop.
    // The dual loop covers indices: base, base+1, base+vec_stride, base+vec_stride+1, ...
    // Uncovered: indices where i is odd relative to the dual-loop pattern.
    // Easiest correct approach: after dual loop, do a single-stride cleanup for any remainder.
    // The dual loop processes pairs starting at base=tid*2 with step vec_stride=stride*2.
    // Remaining float4 elements are those with index >= (N4 / (2*single_stride)) * (2*single_stride)
    // that weren't covered. Use a simple single-stride loop over the tail.
    int64_t dual_covered = ((N4) / (2 * single_stride)) * (2 * single_stride);
    for (int64_t i = dual_covered + tid_global; i < N4; i += single_stride) {
        float4 v = x4[i];
        v.x *= inv_norm; v.y *= inv_norm; v.z *= inv_norm; v.w *= inv_norm;
        x4[i] = v;
    }

    // Scalar tail for elements after N4*4
    int64_t tail_start = N4 * 4;
    for (int64_t i = tail_start + tid_global; i < N; i += single_stride) {
        x[i] *= inv_norm;
    }
}

torch::Tensor frobenius_normalize_cuda(torch::Tensor x) {
    int64_t N = x.numel();
    if (N == 0) {
        return x;
    }

    const int BLOCK = 256;
    int64_t grid = (N + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    auto opts_d = torch::TensorOptions().dtype(torch::kFloat64).device(x.device());
    auto partials = torch::empty({grid}, opts_d);
    auto sum_sq = torch::empty({1}, opts_d);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sum_sq_kernel<<<grid, BLOCK, 0, stream>>>(
        x.data_ptr<float>(),
        partials.data_ptr<double>(),
        N
    );

    int rblock = (int)grid < 1024 ? (int)grid : 1024;
    if (rblock < 1) rblock = 1;
    reduce_partials_kernel<<<1, rblock, 0, stream>>>(
        partials.data_ptr<double>(),
        sum_sq.data_ptr<double>(),
        (int)grid
    );

    int64_t N4 = N / 4;
    // Each thread handles 2 float4 per iteration, so we need ceil(N4 / 2) threads
    int64_t grid2 = (N4 + 2 * BLOCK - 1) / (2 * BLOCK);
    if (grid2 < 1) grid2 = 1;
    if (grid2 > 65535) grid2 = 65535;

    normalize_kernel_f4_inplace<<<grid2, BLOCK, 0, stream>>>(
        x.data_ptr<float>(),
        N4,
        N,
        sum_sq.data_ptr<double>()
    );

    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Frobenius norm normalization.
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the Frobenius norm normalization layer.
                """
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
                    return _stark_get_extension().frobenius_normalize(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        norm = torch.norm(x, p='fro')
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        return x / norm
        # <<<END_IMPROVE>>>
