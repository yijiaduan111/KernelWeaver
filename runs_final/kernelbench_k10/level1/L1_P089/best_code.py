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
    return f'stark_cuda_l1_p89_{digest}'

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

torch::Tensor cumsum_lastdim_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_lastdim", &cumsum_lastdim_cuda, "Contiguous last-dim cumsum (CUDA float32)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_inclusive_scan_tile(float val, int lane) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

// Short-row kernel: each warp handles one row, 4 warps per block
__global__ void cumsum_lastdim_warp_packed_kernel(const float* __restrict__ input,
                                                   float* __restrict__ output,
                                                   int cols, int rows) {
    const int warps_per_block = 4;
    int warp_id_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * warps_per_block + warp_id_in_block;
    if (row >= rows) return;

    const float* row_in = input + (long long)row * cols;
    float* row_out = output + (long long)row * cols;

    float val = (lane < cols) ? row_in[lane] : 0.0f;
    float scanned = warp_inclusive_scan_tile(val, lane);
    if (lane < cols) {
        row_out[lane] = scanned;
    }
}

template<int BLOCK_THREADS>
__global__ void cumsum_lastdim_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int cols) {
    constexpr int NUM_WARPS = (BLOCK_THREADS + 31) / 32;
    __shared__ float warp_sums[NUM_WARPS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    const float* row_in = input + (long long)row * cols;
    float* row_out = output + (long long)row * cols;

    float tile_prefix = 0.0f;

    for (int base = 0; base < cols; base += BLOCK_THREADS) {
        int idx = base + tid;
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        float scanned = warp_inclusive_scan_tile(val, lane);

        if (lane == 31) {
            warp_sums[warp_id] = scanned;
        }
        int last_tid_in_tile = min(BLOCK_THREADS, cols - base) - 1;
        if (tid == last_tid_in_tile && lane != 31) {
            warp_sums[warp_id] = scanned;
        }
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
            for (int offset = 1; offset < 32; offset <<= 1) {
                float n = __shfl_up_sync(0xffffffff, v, offset);
                if (lane >= offset) v += n;
            }
            if (lane < NUM_WARPS) warp_sums[lane] = v;
        }
        __syncthreads();

        float warp_prefix = (warp_id > 0) ? warp_sums[warp_id - 1] : 0.0f;
        float result = tile_prefix + warp_prefix + scanned;

        if (idx < cols) {
            row_out[idx] = result;
        }

        int last_warp = (min(BLOCK_THREADS, cols - base) - 1) >> 5;
        tile_prefix += warp_sums[last_warp];
        __syncthreads();
    }
}

torch::Tensor cumsum_lastdim_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 1, "Input must have at least 1 dimension");

    auto output = torch::empty_like(x);
    int cols = x.size(-1);
    long long rows = x.numel() / cols;

    if (rows == 0 || cols == 0) return output;

    if (cols <= 32) {
        // Packed warp kernel: 4 warps per block, each warp handles one row
        const int warps_per_block = 4;
        int grid = ((int)rows + warps_per_block - 1) / warps_per_block;
        cumsum_lastdim_warp_packed_kernel<<<grid, warps_per_block * 32>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), cols, (int)rows);
    } else if (cols <= 64) {
        constexpr int BT = 64;
        size_t smem = ((BT + 31) / 32) * sizeof(float);
        cumsum_lastdim_kernel<BT><<<(int)rows, BT, smem>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), cols);
    } else if (cols <= 192) {
        constexpr int BT = 128;
        size_t smem = ((BT + 31) / 32) * sizeof(float);
        cumsum_lastdim_kernel<BT><<<(int)rows, BT, smem>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), cols);
    } else if (cols <= 512) {
        constexpr int BT = 256;
        size_t smem = ((BT + 31) / 32) * sizeof(float);
        cumsum_lastdim_kernel<BT><<<(int)rows, BT, smem>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), cols);
    } else {
        constexpr int BT = 512;
        size_t smem = ((BT + 31) / 32) * sizeof(float);
        cumsum_lastdim_kernel<BT><<<(int)rows, BT, smem>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), cols);
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

        Parameters:
            dim (int): The dimension along which to perform the scan operation.
        """
    def __init__(self, dim):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initialize the Scan model.

                Args:
                    dim (int): The dimension along which to perform the cumulative sum.
                """
        self.dim = dim
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape), where `*input_shape`
            can vary depending on the use case.

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        ndim = x.dim()
        dim = self.dim if self.dim >= 0 else self.dim + ndim
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (x.is_cuda and x.is_contiguous() and x.dtype == torch.float32
                and ndim >= 1 and dim == ndim - 1):
            return _stark_get_extension().cumsum_lastdim(x)
        return torch.cumsum(x, dim=self.dim)
        # <<<END_IMPROVE>>>
