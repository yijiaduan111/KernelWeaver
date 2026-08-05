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
    return f'stark_cuda_l1_p14_{digest}'

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

torch::Tensor upper_triangular_matmul_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor upper_triangular_matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(a.size(0) == a.size(1), "a must be square");
    TORCH_CHECK(b.size(0) == b.size(1), "b must be square");
    TORCH_CHECK(a.size(0) == b.size(0), "a and b must have the same shape");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    return upper_triangular_matmul_cuda(a.contiguous(), b.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upper_triangular_matmul", &upper_triangular_matmul, "Upper triangular matrix multiplication (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size: 32x32 output tile, 16x16 threads, each thread computes 2x2 register micro-tile
#define TILE_OUT 32
#define TILE_THREAD 16
#define REG_TILE 2
#define TILE_K 32

__global__ void upper_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    int block_row = blockIdx.y;  // output row tile index
    int block_col = blockIdx.x;  // output col tile index

    // Skip blocks fully below the diagonal
    if (block_row > block_col) return;

    // Each thread computes a 2x2 sub-tile within the 32x32 output tile
    // threadIdx.y in [0,16), threadIdx.x in [0,16)
    // Thread (ty, tx) handles rows [block_row*32 + ty*2, ..., ty*2+1]
    //                          cols [block_col*32 + tx*2, ..., tx*2+1]

    int base_row = block_row * TILE_OUT + threadIdx.y * REG_TILE;
    int base_col = block_col * TILE_OUT + threadIdx.x * REG_TILE;

    float acc[REG_TILE][REG_TILE];
    #pragma unroll
    for (int i = 0; i < REG_TILE; i++)
        #pragma unroll
        for (int j = 0; j < REG_TILE; j++)
            acc[i][j] = 0.0f;

    // Shared memory for A and B tiles (32x32 each)
    __shared__ float As[TILE_K][TILE_OUT];  // A tile: rows in output-row-tile, cols in k-tile
    __shared__ float Bs[TILE_K][TILE_OUT];  // B tile: rows in k-tile, cols in output-col-tile

    // K tiles: from block_row to block_col (inclusive)
    int k_start = block_row;
    int k_end   = block_col;

    for (int kb = k_start; kb <= k_end; kb++) {
        // Load A tile: A[block_row*32 : (block_row+1)*32, kb*32 : (kb+1)*32]
        // 16x16 threads load 32x32 tile => each thread loads 2x2 elements
        #pragma unroll
        for (int i = 0; i < REG_TILE; i++) {
            int a_row = block_row * TILE_OUT + threadIdx.y * REG_TILE + i;
            #pragma unroll
            for (int j = 0; j < REG_TILE; j++) {
                int a_col = kb * TILE_OUT + threadIdx.x * REG_TILE + j;
                float val = 0.0f;
                if (a_row < N && a_col < N && a_row <= a_col)
                    val = A[a_row * N + a_col];
                As[threadIdx.y * REG_TILE + i][threadIdx.x * REG_TILE + j] = val;
            }
        }

        // Load B tile: B[kb*32 : (kb+1)*32, block_col*32 : (block_col+1)*32]
        #pragma unroll
        for (int i = 0; i < REG_TILE; i++) {
            int b_row = kb * TILE_OUT + threadIdx.y * REG_TILE + i;
            #pragma unroll
            for (int j = 0; j < REG_TILE; j++) {
                int b_col = block_col * TILE_OUT + threadIdx.x * REG_TILE + j;
                float val = 0.0f;
                if (b_row < N && b_col < N && b_row <= b_col)
                    val = B[b_row * N + b_col];
                Bs[threadIdx.y * REG_TILE + i][threadIdx.x * REG_TILE + j] = val;
            }
        }

        __syncthreads();

        // Accumulate: each thread computes its 2x2 output patch
        #pragma unroll
        for (int i = 0; i < REG_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < REG_TILE; j++) {
                float sum = 0.0f;
                int a_row_local = threadIdx.y * REG_TILE + i;
                int b_col_local = threadIdx.x * REG_TILE + j;
                #pragma unroll
                for (int t = 0; t < TILE_K; t++) {
                    sum += As[a_row_local][t] * Bs[t][b_col_local];
                }
                acc[i][j] += sum;
            }
        }

        __syncthreads();
    }

    // Write results back
    #pragma unroll
    for (int i = 0; i < REG_TILE; i++) {
        int row = base_row + i;
        #pragma unroll
        for (int j = 0; j < REG_TILE; j++) {
            int col = base_col + j;
            if (row < N && col < N && row <= col) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

torch::Tensor upper_triangular_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    int N = a.size(0);
    auto c = torch::zeros_like(a);

    dim3 block(TILE_THREAD, TILE_THREAD);
    int grid_dim = (N + TILE_OUT - 1) / TILE_OUT;
    dim3 grid(grid_dim, grid_dim);

    upper_triangular_matmul_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N
    );

    return c;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
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
                Performs matrix multiplication for upper triangular matrices.

                Args:
                    A (torch.Tensor): Upper triangular matrix of shape (N, N).
                    B (torch.Tensor): Upper triangular matrix of shape (N, N).

                Returns:
                    torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (
        A.is_cuda and B.is_cuda and
        A.dtype == torch.float32 and B.dtype == torch.float32 and
        A.dim() == 2 and B.dim() == 2 and
        A.size(0) == A.size(1) and
        B.size(0) == B.size(1) and
        A.size(0) == B.size(0) and
        A.is_contiguous() and B.is_contiguous() and
        A.size(0) >= 64
        ):
            return _stark_get_extension().upper_triangular_matmul(A, B)
        return torch.triu(torch.matmul(A, B))
        # <<<END_IMPROVE>>>
