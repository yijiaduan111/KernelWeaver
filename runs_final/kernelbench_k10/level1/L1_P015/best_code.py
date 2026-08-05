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
    return f'stark_cuda_l1_p15_{digest}'

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

torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lower_triangular_matmul", &lower_triangular_matmul_cuda, "Lower triangular matrix multiplication (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void lower_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int brow = blockIdx.y;
    int bcol = blockIdx.x;

    // Skip upper triangular blocks
    if (bcol > brow) return;

    int row = brow * TILE_SIZE + threadIdx.y;
    int col = bcol * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Diagonal block: bcol == brow, only one tile, always boundary
    if (bcol == brow) {
        int a_col = bcol * TILE_SIZE + threadIdx.x;
        int b_row = bcol * TILE_SIZE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < N && a_col < N && row >= a_col) ? A[row * N + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < N && col < N && b_row >= col) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    } else {
        // First boundary tile: kt == bcol
        {
            int kt = bcol;
            int a_col = kt * TILE_SIZE + threadIdx.x;
            int b_row = kt * TILE_SIZE + threadIdx.y;
            // For A tile: row is in brow block (strictly below bcol diagonal), a_col is in bcol block.
            // row >= a_col check needed only for the first tile (bcol tile of A)
            sA[threadIdx.y][threadIdx.x] = (row < N && a_col < N && row >= a_col) ? A[row * N + a_col] : 0.0f;
            // For B tile: b_row is in bcol block, col is in bcol block. b_row >= col check needed.
            sB[threadIdx.y][threadIdx.x] = (b_row < N && col < N && b_row >= col) ? B[b_row * N + col] : 0.0f;
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
            }
            __syncthreads();
        }

        // Interior tiles: bcol < kt < brow â no triangular predicates needed
        for (int kt = bcol + 1; kt < brow; ++kt) {
            int a_col = kt * TILE_SIZE + threadIdx.x;
            int b_row = kt * TILE_SIZE + threadIdx.y;
            // Interior: row is in brow (below kt), a_col is in kt (below brow) => row >= a_col always true
            // b_row is in kt (above bcol), col is in bcol (below kt) => b_row >= col always true
            sA[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
            sB[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
            }
            __syncthreads();
        }

        // Last boundary tile: kt == brow
        {
            int kt = brow;
            int a_col = kt * TILE_SIZE + threadIdx.x;
            int b_row = kt * TILE_SIZE + threadIdx.y;
            // For A tile: row is in brow block, a_col is in brow block. row >= a_col check needed.
            sA[threadIdx.y][threadIdx.x] = (row < N && a_col < N && row >= a_col) ? A[row * N + a_col] : 0.0f;
            // For B tile: b_row is in brow block (above bcol), col is in bcol block. b_row >= col always true.
            sB[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
            }
            __syncthreads();
        }
    }

    if (row < N && col < N && row >= col) {
        C[row * N + col] = sum;
    }
}

torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Inputs must have the same shape");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    int grid_dim = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(grid_dim, grid_dim);
    dim3 block(TILE_SIZE, TILE_SIZE);

    lower_triangular_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices. 
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
                Performs matrix multiplication of lower triangular matrices A and B.

                Args:
                    A (torch.Tensor): Lower triangular matrix of shape (N, N).
                    B (torch.Tensor): Lower triangular matrix of shape (N, N).

                Returns:
                    torch.Tensor: The result of matrix multiplication C of shape (N, N).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (A.is_cuda and B.is_cuda and A.dtype == torch.float32 and B.dtype == torch.float32 and A.dim() == 2 and B.dim() == 2 and A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1] and A.shape == B.shape and A.is_contiguous() and B.is_contiguous()):
            N = A.shape[0]
            if N <= 128:
                return _stark_get_extension().lower_triangular_matmul(A, B)
            return torch.matmul(A, B)
        return torch.matmul(A, B)
        # <<<END_IMPROVE>>>
