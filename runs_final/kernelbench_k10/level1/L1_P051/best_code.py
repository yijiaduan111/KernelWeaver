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
    return f'stark_cuda_l1_p51_{digest}'

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

torch::Tensor argmax_dim1_thread_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("argmax_dim1_thread_cuda", &argmax_dim1_thread_cuda, "Thread-per-output coalesced argmax over dim=1 for contiguous float32 3D tensors");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Each thread handles TILE adjacent inner columns.
// Threads in a block cover a contiguous range of inner_idx values -> coalesced loads.
// Grid: gridDim.x = ceil(inner / (BLOCK_X * TILE)), gridDim.y = outer

#define BLOCK_X 256
#define TILE 4

__global__ void __launch_bounds__(BLOCK_X, 6) argmax_dim1_thread_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int outer,
    int reduce,
    int inner
) {
    int base_inner = ((int)blockIdx.x * BLOCK_X + (int)threadIdx.x) * TILE;
    int outer_idx = (int)blockIdx.y;
    if (base_inner >= inner || outer_idx >= outer) return;

    // Compute how many columns this thread actually owns
    int cols = (base_inner + TILE <= inner) ? TILE : (inner - base_inner);

    const float* row_base = input + (int64_t)outer_idx * reduce * inner + base_inner;

    float best_val[TILE];
    int   best_idx[TILE];
    #pragma unroll
    for (int t = 0; t < TILE; ++t) {
        best_val[t] = -FLT_MAX;
        best_idx[t] = 0;
    }

    for (int k = 0; k < reduce; ++k) {
        const float* ptr = row_base + (int64_t)k * inner;
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            if (t < cols) {
                float v = __ldg(ptr + t);
                if (v > best_val[t]) {
                    best_val[t] = v;
                    best_idx[t] = k;
                }
            }
        }
    }

    int64_t* out_base = output + (int64_t)outer_idx * inner + base_inner;
    #pragma unroll
    for (int t = 0; t < TILE; ++t) {
        if (t < cols) {
            out_base[t] = (int64_t)best_idx[t];
        }
    }
}

torch::Tensor argmax_dim1_thread_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");

    int outer  = (int)input.size(0);
    int reduce = (int)input.size(1);
    int inner  = (int)input.size(2);

    auto output = torch::empty({outer, inner},
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));

    int threads_needed = (inner + TILE - 1) / TILE;
    dim3 block(BLOCK_X, 1);
    dim3 grid((threads_needed + BLOCK_X - 1) / BLOCK_X, outer);

    argmax_dim1_thread_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        outer, reduce, inner
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Argmax over a specified dimension.
        """
    def __init__(self, dim: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the model with the dimension to perform argmax.

                Args:
                    dim (int): The dimension to perform argmax over.
                """
        self.dim = dim
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.dim() == 3 and self.dim == 1:
                    return _stark_get_extension().argmax_dim1_thread_cuda(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return torch.argmax(x, dim=self.dim)
        # <<<END_IMPROVE>>>
