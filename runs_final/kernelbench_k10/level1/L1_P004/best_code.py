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
    return f'stark_cuda_l1_p4_{digest}'

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

torch::Tensor matrix_vector_mul(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_vector_mul", &matrix_vector_mul, "Matrix-vector multiply custom CUDA GEMV (streaming A, cached B)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom GEMV kernel: one warp per output row.
// A is loaded with non-temporal (streaming) loads via __ldcs to avoid cache pollution.
// B is loaded with normal cached loads so it stays resident in L2.
// Single-accumulator, no unrolling to minimize register pressure.
__global__ void gemv_streaming_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int K) {
    // One warp per row
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const float* A_row = A + (long long)row * K;
    float sum = 0.0f;

    int lane = threadIdx.x; // 0..31

    // Vectorized loop: process 4 floats at a time using float4
    // Use __ldcs for non-temporal streaming load of A
    int K4 = K / 4;
    const float4* A_row4 = reinterpret_cast<const float4*>(A_row);
    const float4* B4 = reinterpret_cast<const float4*>(B);

    for (int i = lane; i < K4; i += 32) {
        // Non-temporal load for A (streaming, evict-first)
        float4 a4 = __ldcs(A_row4 + i);
        // Normal cached load for B (reused across rows)
        float4 b4 = B4[i];
        sum += a4.x * b4.x + a4.y * b4.y + a4.z * b4.z + a4.w * b4.w;
    }

    // Handle tail elements (K % 4)
    int tail_start = K4 * 4;
    for (int i = tail_start + lane; i < K; i += 32) {
        sum += __ldcs(A_row + i) * B[i];
    }

    // Warp reduction
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if (lane == 0) {
        C[row] = sum;
    }
}

torch::Tensor matrix_vector_mul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2 && B.size(1) == 1, "B must be shape (K, 1)");
    TORCH_CHECK(A.size(1) == B.size(0), "Dimension mismatch");

    A = A.contiguous();
    B = B.contiguous();

    int M = (int)A.size(0);
    int K = (int)A.size(1);

    auto C = torch::zeros({M, 1}, A.options());

    // One warp per row; pack multiple warps per block
    const int WARPS_PER_BLOCK = 4;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    gemv_streaming_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K
    );

    return C;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs matrix-vector multiplication (C = A * B).
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if (A.is_cuda and B.is_cuda and
                        A.dtype == torch.float32 and B.dtype == torch.float32 and
                        A.dim() == 2 and B.dim() == 2 and B.size(1) == 1 and
                        A.size(1) == B.size(0)):
                    return _stark_get_extension().matrix_vector_mul(A, B)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return torch.matmul(A, B)
        # <<<END_IMPROVE>>>
