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
    return f'stark_cuda_l1_p9_{digest}'

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

torch::Tensor tall_skinny_matmul_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tall_skinny_matmul_cuda", &tall_skinny_matmul_cuda, "Tall-skinny matmul via cuBLAS");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tiled GEMM kernel for tall-skinny shapes.
// BLOCK_M=64, BLOCK_N=64, K assumed small (<=64).
// Each thread block computes a (BLOCK_M x BLOCK_N) output tile.
// Threads: (16, 16) = 256 threads per block.
// Each thread computes a (4 x 4) output fragment.

#define BLOCK_M 64
#define BLOCK_N 64
#define TM 4
#define TN 4
#define THREADS_M (BLOCK_M / TM)  // 16
#define THREADS_N (BLOCK_N / TN)  // 16

__global__ void tall_skinny_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    // Block tile origin
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    // Thread indices within block
    int ty = threadIdx.y;  // 0..15
    int tx = threadIdx.x;  // 0..15

    // Each thread owns a (TM x TN) = (4x4) output fragment
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // Shared memory for A tile: [BLOCK_M][K+1] and B tile: [K][BLOCK_N+1]
    // K is at most 64; we use dynamic sizing via template or just cap at 64.
    __shared__ float As[BLOCK_M][33];  // K<=32, pad to avoid bank conflicts
    __shared__ float Bs[32][BLOCK_N+1];

    // Load A tile: BLOCK_M rows x K cols
    // 256 threads load BLOCK_M*K = 64*32 = 2048 elements -> 8 elements per thread
    int tid = ty * THREADS_N + tx;  // 0..255
    #pragma unroll
    for (int idx = tid; idx < BLOCK_M * K; idx += THREADS_M * THREADS_N) {
        int r = idx / K;
        int c = idx % K;
        int global_row = block_row + r;
        float val = (global_row < M) ? __ldg(&A[global_row * K + c]) : 0.0f;
        As[r][c] = val;
    }

    // Load B tile: K rows x BLOCK_N cols
    #pragma unroll
    for (int idx = tid; idx < K * BLOCK_N; idx += THREADS_M * THREADS_N) {
        int r = idx / BLOCK_N;
        int c = idx % BLOCK_N;
        int global_col = block_col + c;
        float val = (global_col < N) ? __ldg(&B[r * N + global_col]) : 0.0f;
        Bs[r][c] = val;
    }

    __syncthreads();

    // Compute: each thread computes TM x TN output
    #pragma unroll
    for (int k = 0; k < K; k++) {
        float a_reg[TM];
        float b_reg[TN];
        #pragma unroll
        for (int i = 0; i < TM; i++)
            a_reg[i] = As[ty * TM + i][k];
        #pragma unroll
        for (int j = 0; j < TN; j++)
            b_reg[j] = Bs[k][tx * TN + j];
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++)
                acc[i][j] += a_reg[i] * b_reg[j];
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int global_row = block_row + ty * TM + i;
        if (global_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int global_col = block_col + tx * TN + j;
            if (global_col < N)
                C[global_row * N + global_col] = acc[i][j];
        }
    }
}

torch::Tensor tall_skinny_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Both tensors must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Both tensors must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Both tensors must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match");

    a = a.contiguous();
    b = b.contiguous();

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    // Only use custom kernel for small K (fits in shared memory)
    if (K > 32) {
        return torch::matmul(a, b);
    }

    auto out = torch::empty({M, N}, a.options());

    dim3 block(THREADS_N, THREADS_M);  // (16, 16)
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    tall_skinny_gemm_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        (int)M, (int)K, (int)N
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, A, B):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the matrix multiplication.

                Args:
                    A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
                    B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

                Returns:
                    torch.Tensor: Output matrix of shape (M, N) or (N, M)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (A.is_cuda and B.is_cuda and A.dtype == torch.float32 and B.dtype == torch.float32 and A.dim() == 2 and B.dim() == 2 and A.shape[1] == B.shape[0]):
            return _stark_get_extension().tall_skinny_matmul_cuda(A, B)
        return torch.matmul(A, B)
        # <<<END_IMPROVE>>>
