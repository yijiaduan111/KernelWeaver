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
    return f'stark_cuda_l2_p89_{digest}'

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

torch::Tensor tail_fused_cuda(torch::Tensor x, torch::Tensor subtract);

torch::Tensor tail_fused(torch::Tensor x, torch::Tensor subtract) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(subtract.is_cuda(), "subtract must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N, C, D, H, W)");
    TORCH_CHECK(subtract.dim() == 1, "subtract must be 1D (C)");
    TORCH_CHECK(x.size(1) == subtract.size(0), "channel dimension mismatch");
    return tail_fused_cuda(x.contiguous(), subtract.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tail_fused", &tail_fused, "Fused softmax-subtract-swish-max tail");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Warp-only kernel for C <= 32: each warp processes one voxel,
// all channel values stay in registers, shuffles replace __syncthreads.
__global__ void tail_fused_warp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ subtract,
    float* __restrict__ out,
    int N, int C, int D, int H, int W
) {
    // Each warp handles one voxel. gridDim.x * (blockDim.x/32) warps total.
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x & 31;

    int spatial_size  = D * H * W;
    int total_voxels  = N * spatial_size;
    if (warp_id >= total_voxels) return;

    int n   = warp_id / spatial_size;
    int rem = warp_id % spatial_size;
    int d   = rem / (H * W);
    int hw  = rem % (H * W);
    int h   = hw / W;
    int w   = hw % W;

    // Each lane loads one channel; lanes >= C hold -FLT_MAX / 0.
    float val = -FLT_MAX;
    if (lane < C) {
        int offset = ((n * C + lane) * D + d) * H * W + h * W + w;
        val = __ldg(&x[offset]);
    }

    // --- Warp max for softmax stability ---
    float wmax = val;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        wmax = fmaxf(wmax, __shfl_down_sync(0xffffffff, wmax, delta));
    wmax = __shfl_sync(0xffffffff, wmax, 0);

    // --- exp(v - max) and warp sum ---
    float e = (lane < C) ? expf(val - wmax) : 0.0f;
    float wsum = e;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        wsum += __shfl_down_sync(0xffffffff, wsum, delta);
    wsum = __shfl_sync(0xffffffff, wsum, 0);

    // --- softmax -> subtract -> swish -> warp max ---
    float final_val = -FLT_MAX;
    if (lane < C) {
        float softmax_val  = e / wsum;
        float sub_val      = __ldg(&subtract[lane]);
        float after_sub    = softmax_val - sub_val;
        float swish_val    = after_sub / (1.0f + expf(-after_sub));
        final_val = swish_val;
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        final_val = fmaxf(final_val, __shfl_down_sync(0xffffffff, final_val, delta));

    if (lane == 0) {
        out[warp_id] = final_val;
    }
}

// Generic kernel for C > 32: one block per voxel, shared-memory reductions.
__global__ void tail_fused_generic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ subtract,
    float* __restrict__ out,
    int N, int C, int D, int H, int W
) {
    int voxel       = blockIdx.x;
    int spatial_size = D * H * W;
    int total_voxels = N * spatial_size;
    if (voxel >= total_voxels) return;

    int n   = voxel / spatial_size;
    int rem = voxel % spatial_size;
    int d   = rem / (H * W);
    int hw  = rem % (H * W);
    int h   = hw / W;
    int w   = hw % W;

    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ float smem[];
    float* ch_vals = smem;           // C floats
    float* scratch = smem + C;       // nthreads floats

    for (int c = tid; c < C; c += nthreads) {
        int offset = ((n * C + c) * D + d) * H * W + h * W + w;
        ch_vals[c] = __ldg(&x[offset]);
    }
    __syncthreads();

    // Pass 1: block max
    float local_max = -FLT_MAX;
    for (int c = tid; c < C; c += nthreads)
        local_max = fmaxf(local_max, ch_vals[c]);
    scratch[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
        __syncthreads();
    }
    float global_max = scratch[0];
    __syncthreads();

    // Pass 2: exp + sum
    float local_sum = 0.0f;
    for (int c = tid; c < C; c += nthreads) {
        float e = expf(ch_vals[c] - global_max);
        ch_vals[c] = e;
        local_sum += e;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float global_sum = scratch[0];
    __syncthreads();

    // Pass 3: normalize, subtract, swish, max
    float final_max = -FLT_MAX;
    for (int c = tid; c < C; c += nthreads) {
        float softmax_val = ch_vals[c] / global_sum;
        float after_sub   = softmax_val - __ldg(&subtract[c]);
        float swish_val   = after_sub / (1.0f + expf(-after_sub));
        final_max = fmaxf(final_max, swish_val);
    }
    scratch[tid] = final_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        out[voxel] = scratch[0];
}

torch::Tensor tail_fused_cuda(torch::Tensor x, torch::Tensor subtract) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tail_fused_cuda: only float32 supported");

    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    int spatial_size  = D * H * W;
    int total_voxels  = N * spatial_size;

    auto out = torch::empty({N, D, H, W}, x.options());

    if (C <= 32) {
        // Warp-only path: pack multiple warps per block to improve occupancy.
        // Each warp = 32 threads handles one voxel.
        const int WARPS_PER_BLOCK = 8;
        const int THREADS = WARPS_PER_BLOCK * 32;
        int blocks = (total_voxels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        tail_fused_warp_kernel<<<blocks, THREADS>>>(
            x.data_ptr<float>(),
            subtract.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, D, H, W
        );
    } else {
        // Generic path for larger C.
        int threads = 1;
        while (threads < C) threads <<= 1;
        if (threads > 256) threads = 256;
        if (threads < 32)  threads = 32;
        int smem_size = (C + threads) * sizeof(float);
        tail_fused_generic_kernel<<<total_voxels, threads, smem_size>>>(
            x.data_ptr<float>(),
            subtract.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, D, H, W
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a sequence of operations:
            - ConvTranspose3d
            - MaxPool3d
            - Softmax
            - Subtract
            - Swish
            - Max
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.max_pool(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if x.is_cuda and x.dtype == torch.float32 and self.subtract.is_cuda:
            x = _stark_get_extension().tail_fused(x, self.subtract)
        else:
            x = torch.softmax(x, dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if x.dim() == 5:
                    x = x - self.subtract.view(1, -1, 1, 1, 1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if x.dim() == 5:
                    x = torch.sigmoid(x) * x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        if x.dim() == 5:
                    x = torch.max(x, dim=1)[0]
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
