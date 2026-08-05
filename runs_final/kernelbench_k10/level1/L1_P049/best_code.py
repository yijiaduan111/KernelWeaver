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
    return f'stark_cuda_l1_p49_{digest}'

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

torch::Tensor max_reduce_dim1_tiled_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce_dim1_tiled_cuda", &max_reduce_dim1_tiled_cuda, "max reduce dim=1 tiled cuda");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// Each thread owns COLS_PER_THREAD adjacent output columns.
// Threads in a warp access adjacent K positions -> coalesced loads.
// No shared memory needed: each thread independently reduces over M.
template <int BLOCK_SIZE, int COLS_PER_THREAD>
__global__ void max_reduce_dim1_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int M, int K
) {
    // Each thread handles COLS_PER_THREAD consecutive k columns
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int n   = blockIdx.y;

    // Base k index for this thread
    const int k_base = tid * COLS_PER_THREAD;

    if (n >= N || k_base >= K) return;

    // Register array for partial maxima
    float local_max[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
        local_max[c] = -FLT_MAX;
    }

    const float* batch_base = input + (long long)n * M * K;

    // Loop over reduction dimension M
    for (int m = 0; m < M; m++) {
        const float* row = batch_base + (long long)m * K + k_base;
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; c++) {
            int k = k_base + c;
            if (k < K) {
                float v = row[c];
                local_max[c] = fmaxf(local_max[c], v);
            }
        }
    }

    // Write results
    float* out_base = output + (long long)n * K + k_base;
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
        int k = k_base + c;
        if (k < K) {
            out_base[c] = local_max[c];
        }
    }
}

torch::Tensor max_reduce_dim1_tiled_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");

    const int N = x.size(0);
    const int M = x.size(1);
    const int K = x.size(2);

    auto output = torch::empty({N, K}, x.options());

    const int BLOCK_SIZE = 128;
    const int COLS_PER_THREAD = 8;
    // Number of threads needed to cover K columns
    const int threads_per_row = (K + COLS_PER_THREAD - 1) / COLS_PER_THREAD;
    const int grid_x = (threads_per_row + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(grid_x, N);
    dim3 block(BLOCK_SIZE);

    max_reduce_dim1_tiled_kernel<128, 8><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M, K
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Max reduction over a specific dimension.
        """
    def __init__(self, dim: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the model with the dimension to reduce over.

                Args:
                    dim (int): The dimension to reduce over.
                """
        self.dim = dim
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies Max reduction over the specified dimension to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor.

                Returns:
                    torch.Tensor: Output tensor after Max reduction over the specified dimension.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 3 and self.dim == 1 and x.is_contiguous():
            return _stark_get_extension().max_reduce_dim1_tiled_cuda(x)
        return torch.max(x, dim=self.dim)[0]
        # <<<END_IMPROVE>>>
