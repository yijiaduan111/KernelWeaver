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
    return f'stark_cuda_l1_p7_{digest}'

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

torch::Tensor small_k_matmul_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("small_k_matmul", &small_k_matmul_cuda, "Small-K matmul CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tiled GEMM kernel specialized for small K (K=64).
// Champion tile: BM=64, BN=64, BK=64, TM=4, TN=4, 256 threads/block.
// This version uses float4 vectorized loads for A and B tile staging,
// and a float4 shared-memory load for the B fragment in the compute loop.

#define BM 64
#define BN 64
#define BK 64
#define TM 4
#define TN 4
#define THREADS_M (BM / TM)  // 16
#define THREADS_N (BN / TN)  // 16
#define BLOCK_THREADS (THREADS_M * THREADS_N)  // 256

__global__ void small_k_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N
) {
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = threadIdx.x;
    int thread_row = tid / THREADS_N;
    int thread_col = tid % THREADS_N;

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float acc[TM][TN] = {};

    // Load A tile: BM*BK = 4096 floats, 256 threads -> 16 floats each = 4 float4 each
    // A is (M, 64) contiguous, so row stride = BK = 64.
    {
        const float4* A4 = reinterpret_cast<const float4*>(A);
        float4* sA4 = reinterpret_cast<float4*>(&sA[0][0]);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx4 = tid + i * BLOCK_THREADS;  // 0..1023
            int flat = idx4 * 4;
            int r = flat / BK;
            int c4 = (flat % BK) / 4;
            int global_r = block_row + r;
            if (global_r < M) {
                sA4[r * (BK/4) + c4] = A4[global_r * (BK/4) + c4];
            } else {
                sA4[r * (BK/4) + c4] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }

    // Load B tile: BK*BN = 4096 floats, 256 threads -> 16 floats each = 4 float4 each
    // B is (64, N) contiguous, row stride = N.
    {
        bool full_n = (block_col + BN <= N);
        if (full_n) {
            const float4* B4 = reinterpret_cast<const float4*>(B);
            float4* sB4 = reinterpret_cast<float4*>(&sB[0][0]);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx4 = tid + i * BLOCK_THREADS;
                int flat = idx4 * 4;
                int r = flat / BN;
                int c4 = (flat % BN) / 4;
                int global_c4 = block_col / 4 + c4;
                sB4[r * (BN/4) + c4] = B4[r * (N/4) + global_c4];
            }
        } else {
            int b_elems = BK * BN;
            for (int i = tid; i < b_elems; i += BLOCK_THREADS) {
                int r = i / BN;
                int c = i % BN;
                int global_c = block_col + c;
                sB[r][c] = (global_c < N) ? B[r * N + global_c] : 0.0f;
            }
        }
    }

    __syncthreads();

    // Compute TM x TN output fragment.
    // Load the contiguous TN=4 B values for each k as a single float4 from shared memory.
    int b_col_base = thread_col * TN;  // always a multiple of 4 since TN=4
    #pragma unroll
    for (int k = 0; k < BK; k++) {
        float a_vals[TM];
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            a_vals[m] = sA[thread_row * TM + m][k];
        }
        // Single float4 load for the 4 contiguous B values at row k, cols [b_col_base .. b_col_base+3]
        float4 bv = *reinterpret_cast<const float4*>(&sB[k][b_col_base]);
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            acc[m][0] += a_vals[m] * bv.x;
            acc[m][1] += a_vals[m] * bv.y;
            acc[m][2] += a_vals[m] * bv.z;
            acc[m][3] += a_vals[m] * bv.w;
        }
    }

    // Write results (scalar, boundary-safe)
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int global_r = block_row + thread_row * TM + m;
        if (global_r >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int global_c = block_col + thread_col * TN + n;
            if (global_c < N) {
                C[global_r * N + global_c] = acc[m][n];
            }
        }
    }
}

torch::Tensor small_k_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    TORCH_CHECK(K == 64 && b.size(0) == 64, "K must be 64");

    auto out = torch::empty({M, N}, a.options());

    dim3 block(BLOCK_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    small_k_matmul_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs matrix multiplication.

                Args:
                    A: Input tensor of shape (M, K).
                    B: Input tensor of shape (K, N).

                Returns:
                    Output tensor of shape (M, N).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (A.is_cuda and B.is_cuda and
                A.dtype == torch.float32 and B.dtype == torch.float32 and
                A.dim() == 2 and B.dim() == 2 and
                A.is_contiguous() and B.is_contiguous() and
                A.shape[1] == 64 and B.shape[0] == 64):
            return _stark_get_extension().small_k_matmul(A, B)
        return torch.matmul(A, B)
        # <<<END_IMPROVE>>>
