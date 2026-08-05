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
    return f'stark_cuda_l1_p47_{digest}'

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

torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduce_dim1", &sum_reduce_dim1_cuda, "sum reduction over dim=1 (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ITEMS_PER_THREAD 4
#define BLOCK_SIZE 128

// Each thread handles ITEMS_PER_THREAD adjacent k outputs for one batch row n.
// Grid: (ceil(K / (BLOCK_SIZE * ITEMS_PER_THREAD)), N)
__global__ void sum_reduce_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int J, int K
) {
    int n = blockIdx.y;
    int k_base = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * ITEMS_PER_THREAD;
    if (n >= N) return;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    bool v0 = (k_base + 0) < K;
    bool v1 = (k_base + 1) < K;
    bool v2 = (k_base + 2) < K;
    bool v3 = (k_base + 3) < K;

    const float* base = input + (long long)n * J * K;
    for (int j = 0; j < J; ++j) {
        long long off = (long long)j * K + k_base;
        if (v0) acc0 += base[off + 0];
        if (v1) acc1 += base[off + 1];
        if (v2) acc2 += base[off + 2];
        if (v3) acc3 += base[off + 3];
    }

    float* out = output + (long long)n * K;
    if (v0) out[k_base + 0] = acc0;
    if (v1) out[k_base + 1] = acc1;
    if (v2) out[k_base + 2] = acc2;
    if (v3) out[k_base + 3] = acc3;
}

torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be rank-3");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int N = (int)x.size(0);
    int J = (int)x.size(1);
    int K = (int)x.size(2);

    auto output_flat = torch::empty({N, K}, x.options());

    int grid_x = (K + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    dim3 grid(grid_x, N);
    dim3 block(BLOCK_SIZE);

    sum_reduce_dim1_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output_flat.data_ptr<float>(),
        N, J, K
    );

    return output_flat.view({N, 1, K});
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs sum reduction over a specified dimension.
        """
    def __init__(self, dim: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the model with the dimension to reduce over.

                Args:
                    dim (int): Dimension to reduce over.
                """
        self.dim = dim
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies sum reduction over the specified dimension.

                Args:
                    x (torch.Tensor): Input tensor of shape (..., dim, ...).

                Returns:
                    torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and self.dim == 1 and x.dtype == torch.float32 and x.dim() == 3 and x.is_contiguous():
            return _stark_get_extension().sum_reduce_dim1(x)
        return torch.sum(x, dim=self.dim, keepdim=True)
        # <<<END_IMPROVE>>>
