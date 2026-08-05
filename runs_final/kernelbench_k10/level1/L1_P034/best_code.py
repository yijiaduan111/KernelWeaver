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
    return f'stark_cuda_l1_p34_{digest}'

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

torch::Tensor instancenorm_inplace(torch::Tensor x, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("instancenorm_inplace", &instancenorm_inplace, "In-place InstanceNorm2d (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// Specialized kernel: spatial must be divisible by 4 and pointer must be 16-byte aligned.
// Uses pure float4 paths with no scalar tail. 512 threads per block for better latency hiding.
__global__ void instancenorm_aligned_kernel(float* __restrict__ x, int nc, int vec_spatial, float eps) {
    int slice = blockIdx.x;
    if (slice >= nc) return;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int base_vec = slice * vec_spatial;  // index into float4 array

    __shared__ float warp_sum[16];
    __shared__ float warp_sq[16];
    __shared__ float mean_val;
    __shared__ float inv_std_val;

    float local_sum = 0.0f;
    float local_sq  = 0.0f;

    float4* xv = reinterpret_cast<float4*>(x) + base_vec;

    // Vectorized accumulation pass - no tail
    for (int i = tid; i < vec_spatial; i += blockDim.x) {
        float4 v = __ldg(xv + i);
        local_sum += v.x + v.y + v.z + v.w;
        local_sq  += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq  += __shfl_down_sync(0xffffffff, local_sq,  offset);
    }

    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
        warp_sq[warp_id]  = local_sq;
    }
    __syncthreads();

    // Final reduction across 16 warps (512 threads / 32 = 16 warps)
    if (warp_id == 0) {
        float w_sum = (lane < 16) ? warp_sum[lane] : 0.0f;
        float w_sq  = (lane < 16) ? warp_sq[lane]  : 0.0f;
        for (int offset = 8; offset > 0; offset >>= 1) {
            w_sum += __shfl_down_sync(0xffffffff, w_sum, offset);
            w_sq  += __shfl_down_sync(0xffffffff, w_sq,  offset);
        }
        if (lane == 0) {
            int spatial = vec_spatial * 4;
            float m   = w_sum / static_cast<float>(spatial);
            float var = w_sq  / static_cast<float>(spatial) - m * m;
            if (var < 0.0f) var = 0.0f;
            mean_val    = m;
            inv_std_val = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    float m   = mean_val;
    float isd = inv_std_val;

    // Vectorized writeback pass - no tail
    for (int i = tid; i < vec_spatial; i += blockDim.x) {
        float4 v = xv[i];
        v.x = (v.x - m) * isd;
        v.y = (v.y - m) * isd;
        v.z = (v.z - m) * isd;
        v.w = (v.w - m) * isd;
        xv[i] = v;
    }
}

// Generic fallback kernel with float4 vectorization + scalar tail. 512 threads per block.
__global__ void instancenorm_inplace_kernel(float* __restrict__ x, int nc, int spatial, float eps) {
    int slice = blockIdx.x;
    if (slice >= nc) return;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int base    = slice * spatial;

    __shared__ float warp_sum[16];
    __shared__ float warp_sq[16];
    __shared__ float mean_val;
    __shared__ float inv_std_val;

    float local_sum = 0.0f;
    float local_sq  = 0.0f;

    int vec_spatial = spatial >> 2;
    int tail_start  = vec_spatial << 2;

    const float4* xv = reinterpret_cast<const float4*>(x + base);
    for (int i = tid; i < vec_spatial; i += blockDim.x) {
        float4 v = __ldg(xv + i);
        local_sum += v.x + v.y + v.z + v.w;
        local_sq  += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int i = tail_start + tid; i < spatial; i += blockDim.x) {
        float v = x[base + i];
        local_sum += v;
        local_sq  += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq  += __shfl_down_sync(0xffffffff, local_sq,  offset);
    }

    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
        warp_sq[warp_id]  = local_sq;
    }
    __syncthreads();

    // Final reduction across 16 warps (512 threads / 32 = 16 warps)
    if (warp_id == 0) {
        float w_sum = (lane < 16) ? warp_sum[lane] : 0.0f;
        float w_sq  = (lane < 16) ? warp_sq[lane]  : 0.0f;
        for (int offset = 8; offset > 0; offset >>= 1) {
            w_sum += __shfl_down_sync(0xffffffff, w_sum, offset);
            w_sq  += __shfl_down_sync(0xffffffff, w_sq,  offset);
        }
        if (lane == 0) {
            float m   = w_sum / static_cast<float>(spatial);
            float var = w_sq  / static_cast<float>(spatial) - m * m;
            if (var < 0.0f) var = 0.0f;
            mean_val    = m;
            inv_std_val = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    float m   = mean_val;
    float isd = inv_std_val;

    float4* xvo = reinterpret_cast<float4*>(x + base);
    for (int i = tid; i < vec_spatial; i += blockDim.x) {
        float4 v = xvo[i];
        v.x = (v.x - m) * isd;
        v.y = (v.y - m) * isd;
        v.z = (v.z - m) * isd;
        v.w = (v.w - m) * isd;
        xvo[i] = v;
    }
    for (int i = tail_start + tid; i < spatial; i += blockDim.x) {
        x[base + i] = (x[base + i] - m) * isd;
    }
}

}  // namespace

torch::Tensor instancenorm_inplace(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int n       = static_cast<int>(x.size(0));
    const int c       = static_cast<int>(x.size(1));
    const int h       = static_cast<int>(x.size(2));
    const int w       = static_cast<int>(x.size(3));
    const int nc      = n * c;
    const int spatial = h * w;
    const float feps  = static_cast<float>(eps);

    // Use specialized aligned no-tail kernel when conditions are met
    bool aligned = (spatial % 4 == 0) &&
                   ((reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16) == 0);

    if (aligned) {
        const int vec_spatial = spatial / 4;
        instancenorm_aligned_kernel<<<nc, 512>>>(x.data_ptr<float>(), nc, vec_spatial, feps);
    } else {
        instancenorm_inplace_kernel<<<nc, 512>>>(x.data_ptr<float>(), nc, spatial, feps);
    }
    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Instance Normalization.
        """
    def __init__(self, num_features: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the InstanceNorm layer.

                Args:
                    num_features (int): Number of features in the input tensor.
                """
        self.inorm = nn.InstanceNorm2d(num_features=num_features)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies Instance Normalization to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

                Returns:
                    torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 4 and x.is_contiguous():
            return _stark_get_extension().instancenorm_inplace(x, float(self.inorm.eps))
        return self.inorm(x)
        # <<<END_IMPROVE>>>
