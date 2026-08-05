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
    return f'stark_cuda_l2_p55_{digest}'

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

torch::Tensor matmul_maxpool_sum_scale_cuda(torch::Tensor x, double scale_factor, int kernel_size);

torch::Tensor matmul_maxpool_sum_scale(torch::Tensor x, double scale_factor, int kernel_size) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    return matmul_maxpool_sum_scale_cuda(x, scale_factor, kernel_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_maxpool_sum_scale", &matmul_maxpool_sum_scale, "Fused maxpool-sum-scale (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one batch row.
// Each thread strides over pool windows, computing max of each window,
// accumulates local sum, then does block reduction.
__global__ void maxpool_sum_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int kernel_size,
    float scale_factor
) {
    int row = blockIdx.x;
    const float* row_ptr = input + row * width;
    int pool_len = width / kernel_size;  // number of complete windows

    float local_sum = 0.0f;

    // Each thread handles pool windows at indices: tid, tid+blockDim.x, ...
    for (int i = threadIdx.x; i < pool_len; i += blockDim.x) {
        int base = i * kernel_size;
        float mx = row_ptr[base];
        for (int k = 1; k < kernel_size; k++) {
            float v = row_ptr[base + k];
            mx = fmaxf(mx, v);
        }
        local_sum += mx;
    }

    // Warp-level reduction
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Block-level reduction via shared memory
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction in first warp
    int num_warps = (blockDim.x + 31) >> 5;
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x == 0) {
            output[row] = val * scale_factor;
        }
    }
}

torch::Tensor matmul_maxpool_sum_scale_cuda(torch::Tensor x, double scale_factor, int kernel_size) {
    TORCH_CHECK(x.size(1) % kernel_size == 0,
        "width must be divisible by kernel_size, got width=", x.size(1), " kernel_size=", kernel_size);

    x = x.contiguous();
    int batch = x.size(0);
    int width = x.size(1);

    auto output = torch::empty({batch}, x.options());

    // Use 256 threads per block; each handles ceil(pool_len/256) windows
    int block_size = 256;
    dim3 grid(batch);
    dim3 block(block_size);

    maxpool_sum_scale_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        width,
        kernel_size,
        (float)scale_factor
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs matrix multiplication, max pooling, sum, and scaling.
        """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_features).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_features).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.matmul(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        ks = self.max_pool.kernel_size if isinstance(self.max_pool.kernel_size, int) else self.max_pool.kernel_size[0]
        self._used_fused = False
        if x.is_cuda and ks >= 1 and x.size(1) % ks == 0:
            x = _stark_get_extension().matmul_maxpool_sum_scale(x, float(self.scale_factor), ks)
            self._used_fused = True
        else:
            x = self.max_pool(x.unsqueeze(1)).squeeze(1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not self._used_fused:
                    x = torch.sum(x, dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if not self._used_fused:
                    x = x * self.scale_factor
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
