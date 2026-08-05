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
    return f'stark_cuda_l2_p65_{digest}'

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

torch::Tensor fused_pool_sigmoid_sum(torch::Tensor conv_out, int64_t pool_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_pool_sigmoid_sum", &fused_pool_sigmoid_sum, "Fused avg pool + sigmoid + sum (NCHW float32)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Helper: horizontal sum of a float4
__device__ __forceinline__ float sum_float4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

// Generic kernel for arbitrary pool_k
__launch_bounds__(256, 4)
__global__ void fused_pool_sigmoid_sum_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ result,
    int N, int C, int H, int W, int pool_k
) {
    int n = blockIdx.x;
    int pH = H / pool_k;
    int pW = W / pool_k;
    float inv_pool = 1.0f / (float)(pool_k * pool_k);
    int spatial = pH * pW;

    float thread_sum = 0.0f;
    for (int c = 0; c < C; c++) {
        const float* plane = conv_out + ((n * C + c) * H) * W;
        for (int linear = threadIdx.x; linear < spatial; linear += blockDim.x) {
            int ph = linear / pW;
            int pw = linear - ph * pW;
            int h0 = ph * pool_k;
            int w0 = pw * pool_k;

            float pool_val =0.0f;
            for (int kh = 0; kh < pool_k; kh++) {
                const float* row = plane + (h0 + kh) * W + w0;
                for (int kw = 0; kw < pool_k; kw++) {
                    pool_val += row[kw];
                }
            }
            float v = pool_val * inv_pool;
            thread_sum += 1.0f / (1.0f + __expf(-v));
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int wid= threadIdx.x >> 5;
    if (lane == 0) smem[wid] = thread_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        float v = (threadIdx.x < 8) ? smem[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (threadIdx.x == 0) result[n] = v;
    }
}

// Specialized kernel for pool_k == 4 with float4-vectorized row loads (aligned fast path)
__launch_bounds__(256, 4)
__global__ void fused_pool4_sigmoid_sum_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ result,
    int N, int C, int H, int W
) {
    int n = blockIdx.x;
    int pH = H >> 2;
    int pW = W >> 2;
    int spatial = pH * pW;
    const float inv16 = 1.0f / 16.0f;

    float thread_sum = 0.0f;
    for (int c = 0; c < C; c++) {
        const float* plane = conv_out + ((n * C + c) * H) * W;
        for (int linear = threadIdx.x; linear < spatial; linear += blockDim.x) {
            int ph = linear / pW;
            int pw = linear - ph * pW;
            int h0 = ph << 2;
            int w0 = pw << 2;

            const float* r0 = plane + (h0 + 0) * W + w0;
            const float* r1 = plane + (h0 + 1) * W + w0;
            const float* r2 = plane + (h0 + 2) * W + w0;
            const float* r3 = plane + (h0 + 3) * W + w0;

            float pool_val;
            // Use vectorized float4 loads when all row pointers are 16-byte aligned
            if (((reinterpret_cast<uintptr_t>(r0) |
                  reinterpret_cast<uintptr_t>(r1) |
                  reinterpret_cast<uintptr_t>(r2) |
                  reinterpret_cast<uintptr_t>(r3)) & 0xFu) == 0u) {
                pool_val = sum_float4(*reinterpret_cast<const float4*>(r0))
                         + sum_float4(*reinterpret_cast<const float4*>(r1))
                         + sum_float4(*reinterpret_cast<const float4*>(r2))
                         + sum_float4(*reinterpret_cast<const float4*>(r3));
            } else {
                pool_val = r0[0] + r0[1] + r0[2] + r0[3]
                         + r1[0] + r1[1] + r1[2] + r1[3]
                         + r2[0] + r2[1] + r2[2] + r2[3]
                         + r3[0] + r3[1] + r3[2] + r3[3];
            }

            float v = pool_val * inv16;
            thread_sum += 1.0f / (1.0f + __expf(-v));
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = thread_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        float v = (threadIdx.x < 8) ? smem[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (threadIdx.x == 0) result[n] = v;
    }
}

torch::Tensor fused_pool_sigmoid_sum(torch::Tensor conv_out, int64_t pool_k) {
    TORCH_CHECK(conv_out.is_cuda(),"conv_out must be a CUDA tensor");
    TORCH_CHECK(conv_out.is_contiguous(), "conv_out must be contiguous");
    TORCH_CHECK(conv_out.scalar_type() == torch::kFloat32, "conv_out must be float32");
    TORCH_CHECK(conv_out.dim() == 4,"conv_out must be 4-D NCHW");

    int N = (int)conv_out.size(0);
    int C = (int)conv_out.size(1);
    int H = (int)conv_out.size(2);
    int W = (int)conv_out.size(3);

    auto result = torch::zeros({N}, conv_out.options());

    if (pool_k == 4) {
        fused_pool4_sigmoid_sum_kernel<<<N, 256>>>(
            conv_out.data_ptr<float>(),
            result.data_ptr<float>(),
            N, C, H, W
        );
    } else {
        fused_pool_sigmoid_sum_kernel<<<N, 256>>>(
            conv_out.data_ptr<float>(),
            result.data_ptr<float>(),
            N, C, H, W, (int)pool_k
        );
    }

    return result;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        This model performs a convolution, average pooling, applies sigmoid, and sums the result.
        """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        pool_k = self.avg_pool.kernel_size
        if isinstance(pool_k, (list, tuple)):
            pool_k = pool_k[0]
        x = _stark_get_extension().fused_pool_sigmoid_sum(x.contiguous(), int(pool_k))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x  # sigmoid fused into CUDA kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = x  # reduction fused into CUDA kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
