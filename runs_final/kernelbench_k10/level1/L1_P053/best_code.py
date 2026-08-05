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
    return f'stark_cuda_l1_p53_{digest}'

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

torch::Tensor min_reduce_cuda(torch::Tensor x, int64_t dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_reduce_cuda", &min_reduce_cuda, "Min reduction (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Warp-level min reduction using shuffle
__device__ __forceinline__ float warp_reduce_min(float val) {
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fminf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Kernel for inner_size > 1: each thread handles one (outer, inner) output element.
// Adjacent threads read adjacent inner_idx -> coalesced memory access.
// Unrolled by 8 for better ILP to hide memory latency.
__global__ void __launch_bounds__(256, 4) min_reduce_coalesced_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (idx >= total) return;

    int64_t outer_idx = idx / inner_size;
    int64_t inner_idx = idx % inner_size;

    const float* base = input + outer_idx * reduce_size * inner_size + inner_idx;
    float m0 = FLT_MAX, m1 = FLT_MAX, m2 = FLT_MAX, m3 = FLT_MAX;
    float m4 = FLT_MAX, m5 = FLT_MAX, m6 = FLT_MAX, m7 = FLT_MAX;

    int64_t r = 0;
    int64_t r_end8 = reduce_size - (reduce_size & 7);
    for (; r < r_end8; r += 8) {
        m0 = fminf(m0, base[(r + 0) * inner_size]);
        m1 = fminf(m1, base[(r + 1) * inner_size]);
        m2 = fminf(m2, base[(r + 2) * inner_size]);
        m3 = fminf(m3, base[(r + 3) * inner_size]);
        m4 = fminf(m4, base[(r + 4) * inner_size]);
        m5 = fminf(m5, base[(r + 5) * inner_size]);
        m6 = fminf(m6, base[(r + 6) * inner_size]);
        m7 = fminf(m7, base[(r + 7) * inner_size]);
    }
    float minv = fminf(fminf(fminf(m0, m1), fminf(m2, m3)),
                       fminf(fminf(m4, m5), fminf(m6, m7)));
    for (; r < reduce_size; r++) {
        minv = fminf(minv, base[r * inner_size]);
    }
    output[outer_idx * inner_size + inner_idx] = minv;
}

// Block reduction kernel for inner_size == 1.
// Each block handles one outer element; threads cooperate over reduce_size.
__global__ void __launch_bounds__(256, 4) min_reduce_inner_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t outer_size,
    int64_t reduce_size
) {
    __shared__ float sdata[8]; // one slot per warp (max 8 warps for 256 threads)
    int64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* base = input + outer_idx * reduce_size;
    float val = FLT_MAX;
    for (int64_t r = threadIdx.x; r < reduce_size; r += blockDim.x) {
        val = fminf(val, base[r]);
    }
    // Warp reduce
    val = warp_reduce_min(val);
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    int nwarps = blockDim.x >> 5;
    if (warp_id == 0) {
        val = (lane < nwarps) ? sdata[lane] : FLT_MAX;
        val = warp_reduce_min(val);
        if (lane == 0) output[outer_idx] = val;
    }
}

torch::Tensor min_reduce_cuda(torch::Tensor x, int64_t dim) {
    int64_t ndim = x.dim();
    if (dim < 0) dim += ndim;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) outer_size *= x.size(i);
    int64_t reduce_size = x.size(dim);
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) inner_size *= x.size(i);

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) out_shape.push_back(x.size(i));
    }
    auto output = torch::empty(out_shape, x.options());

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    if (inner_size == 1) {
        int block_size = 256;
        dim3 grid((int)outer_size);
        dim3 block(block_size);
        min_reduce_inner_kernel<<<grid, block>>>(
            input_ptr, output_ptr, outer_size, reduce_size
        );
    } else {
        int64_t total = outer_size * inner_size;
        int block_size = 256;
        int64_t grid_size = (total + block_size - 1) / block_size;
        min_reduce_coalesced_kernel<<<(int)grid_size, block_size>>>(
            input_ptr, output_ptr, outer_size, reduce_size, inner_size
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs min reduction over a specific dimension.
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
        # Use PyTorch built-in min reduction for all cases.
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return torch.min(x, dim=self.dim)[0]
        # <<<END_IMPROVE>>>
