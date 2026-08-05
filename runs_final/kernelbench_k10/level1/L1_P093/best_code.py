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
    return f'stark_cuda_l1_p93_{digest}'

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

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_cumsum_cuda", &masked_cumsum_cuda, "masked_cumsum_cuda");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Generic kernel: each thread handles one (outer, inner) lane, loops over dim_size sequentially.
// Fuses mask application into the scan loop without materializing x * mask.
template <typename scalar_t>
__global__ void masked_cumsum_generic_kernel(
    const scalar_t* __restrict__ x,
    const bool*     __restrict__ mask,
    scalar_t*       __restrict__ out,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (idx >= total) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    scalar_t running = static_cast<scalar_t>(0);
    for (int64_t d = 0; d < dim_size; ++d) {
        int64_t offset = outer * dim_size * inner_size + d * inner_size + inner;
        scalar_t val = x[offset];
        bool m = mask[offset];
        running += m ? val : static_cast<scalar_t>(0);
        out[offset] = running;
    }
}

// Fast-path kernel for innermost dimension (inner_size == 1), dim_size <= 1024.
// Each block handles one row. Threads cooperate via shared memory for the prefix scan.
template <typename scalar_t>
__global__ void masked_cumsum_innermost_kernel(
    const scalar_t* __restrict__ x,
    const bool*     __restrict__ mask,
    scalar_t*       __restrict__ out,
    int64_t dim_size
) {
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const scalar_t* x_row    = x    + (int64_t)row * dim_size;
    const bool*     mask_row = mask + (int64_t)row * dim_size;
    scalar_t*       out_row  = out  + (int64_t)row * dim_size;

    // Load masked values into shared memory.
    scalar_t v = static_cast<scalar_t>(0);
    if (tid < dim_size) {
        v = mask_row[tid] ? x_row[tid] : static_cast<scalar_t>(0);
    }
    sdata[tid] = v;
    __syncthreads();

    // Inclusive prefix sum via Hillis-Steele (up-sweep).
    for (int stride = 1; stride < block_size; stride <<= 1) {
        scalar_t tmp = static_cast<scalar_t>(0);
        if (tid >= stride && tid < dim_size) {
            tmp = sdata[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < dim_size) {
            sdata[tid] += tmp;
        }
        __syncthreads();
    }

    if (tid < dim_size) {
        out_row[tid] = sdata[tid];
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");

    int64_t ndim = x.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto out = torch::empty_like(x);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= x.size(i);
    int64_t dim_size  = x.size(dim);
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= x.size(i);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", [&] {
        if (inner_size == 1 && dim_size <= 1024) {
            // Fast path: one block per row, shared-memory inclusive scan.
            // Round up block size to next power of two for correctness of Hillis-Steele.
            int bs = 1;
            while (bs < dim_size) bs <<= 1;
            size_t smem_bytes = bs * sizeof(scalar_t);
            masked_cumsum_innermost_kernel<scalar_t><<<outer_size, bs, smem_bytes>>>(
                x.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                out.data_ptr<scalar_t>(),
                dim_size
            );
        } else {
            // Generic path: one thread per (outer, inner) lane.
            int64_t total = outer_size * inner_size;
            int threads = 256;
            int blocks  = (int)((total + threads - 1) / threads);
            masked_cumsum_generic_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                out.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }
    });

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

        Parameters:
            dim (int): The dimension along which to perform the masked cumulative sum.
        """
    def __init__(self, dim):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.dim = dim
        # <<<END_IMPROVE>>>

    def forward(self, x, mask):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
                    mask (torch.Tensor): Boolean mask of the same shape as x.

                Returns:
                    torch.Tensor: Cumulative sum of elements where mask is True.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (x.is_cuda and mask.is_cuda and x.is_contiguous() and mask.is_contiguous()
                and x.shape == mask.shape and mask.dtype == torch.bool
                and x.dtype in (torch.float32, torch.float64)):
            return _stark_get_extension().masked_cumsum_cuda(x, mask, self.dim)
        return torch.cumsum(x * mask, dim=self.dim)
        # <<<END_IMPROVE>>>
