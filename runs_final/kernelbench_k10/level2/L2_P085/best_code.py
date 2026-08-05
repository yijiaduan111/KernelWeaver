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
    return f'stark_cuda_l2_p85_{digest}'

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

torch::Tensor fused_post_ops(
    torch::Tensor x,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor scale,
    int64_t num_groups,
    double eps,
    int64_t pool_k,
    double clamp_min,
    double clamp_max
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_ops", &fused_post_ops, "Fused GroupNorm+Scale+MaxPool+Clamp (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Generic fused kernel: one block per (n, g), shared-memory tree reduction.
// Used as fallback for any shape not matched by the specialized path.
// ---------------------------------------------------------------------------
__global__ void fused_gn_scale_pool_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gn_w,
    const float* __restrict__ gn_b,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int num_groups, float eps,
    int pool_k,
    int pH, int pW,
    float clamp_min, float clamp_max
) {
    const int n = blockIdx.x;
    const int g = blockIdx.y;
    const int cpg = C / num_groups;
    const int c_start = g * cpg;
    const int spatial = H * W;
    const int group_elems = cpg * spatial;

    extern __shared__ float smem[];
    float* s_sum  = smem;
    float* s_sum2 = smem + blockDim.x;

    float local_sum  = 0.0f;
    float local_sum2 = 0.0f;
    for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
        int c  = c_start + i / spatial;
        int hw = i % spatial;
        float v = input[((n * C + c) * H + hw / W) * W + hw % W];
        local_sum  += v;
        local_sum2 += v * v;
    }
    s_sum[threadIdx.x]  = local_sum;
    s_sum2[threadIdx.x] = local_sum2;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x]  += s_sum[threadIdx.x + stride];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean    = s_sum[0]  / (float)group_elems;
    float var     = s_sum2[0] / (float)group_elems - mean * mean;
    float inv_std = rsqrtf(var + eps);

    const int pooled_pixels = pH * pW;
    for (int c_off = 0; c_off < cpg; c_off++) {
        int c  = c_start + c_off;
        float w  = gn_w[c];
        float b  = gn_b[c];
        float sc = scale[c];
        float affine_scale = w * inv_std * sc;
        float affine_bias  = (b - w * mean * inv_std) * sc;

        for (int p = threadIdx.x; p < pooled_pixels; p += blockDim.x) {
            int ph = p / pW;
            int pw = p % pW;
            float max_val = -FLT_MAX;
            for (int kh = 0; kh < pool_k; kh++) {
                int h_in = ph * pool_k + kh;
                if (h_in >= H) continue;
                for (int kw = 0; kw < pool_k; kw++) {
                    int w_in = pw * pool_k + kw;
                    if (w_in >= W) continue;
                    float raw   = input[((n * C + c) * H + h_in) * W + w_in];
                    float normed = raw * affine_scale + affine_bias;
                    if (normed > max_val) max_val = normed;
                }
            }
            max_val = fmaxf(clamp_min, fminf(clamp_max, max_val));
            output[((n * C + c) * pH + ph) * pW + pw] = max_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Specialized kernel: channels_per_group == 4, pool_k == 2.
// One block per (n, g).  Warp-shuffle reduction replaces shared-memory tree.
// The 4-channel loop and 2x2 pool window are fully unrolled.
// Launch with 128 threads (4 warps); smem used only for inter-warp merge.
// ---------------------------------------------------------------------------
__global__ void fused_gn_scale_pool_clamp_k4_p2(
    const float* __restrict__ input,
    const float* __restrict__ gn_w,
    const float* __restrict__ gn_b,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int num_groups, float eps,
    int pH, int pW,
    float clamp_min, float clamp_max
) {
    const int n      = blockIdx.x;
    const int g      = blockIdx.y;
    const int c_start = g * 4;          // cpg == 4
    const int spatial = H * W;          // elements per channel
    const int group_elems = 4 * spatial;

    // --- Phase 1: warp-shuffle reduction for mean/variance ---
    float local_sum  = 0.0f;
    float local_sum2 = 0.0f;
    for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
        int c_off = i / spatial;
        int hw    = i % spatial;
        int h_in  = hw / W;
        int w_in  = hw % W;
        float v = input[((n * C + c_start + c_off) * H + h_in) * W + w_in];
        local_sum  += v;
        local_sum2 += v * v;
    }

    // intra-warp reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum  += __shfl_xor_sync(0xffffffff, local_sum,  mask);
        local_sum2 += __shfl_xor_sync(0xffffffff, local_sum2, mask);
    }

    // inter-warp reduction via shared memory (4 warps max for 128 threads)
    __shared__ float warp_sum[4];
    __shared__ float warp_sum2[4];
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    if (lane == 0) {
        warp_sum[warp_id]  = local_sum;
        warp_sum2[warp_id] = local_sum2;
    }
    __syncthreads();

    // thread 0 accumulates across warps
    float total_sum  = 0.0f;
    float total_sum2 = 0.0f;
    if (threadIdx.x == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nwarps; w++) {
            total_sum  += warp_sum[w];
            total_sum2 += warp_sum2[w];
        }
        warp_sum[0]  = total_sum;
        warp_sum2[0] = total_sum2;
    }
    __syncthreads();
    float mean    = warp_sum[0]  / (float)group_elems;
    float var     = warp_sum2[0] / (float)group_elems - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // --- Phase 2: normalize + affine + scale + maxpool(2x2) + clamp ---
    // Precompute per-channel affine constants for the 4 channels (unrolled).
    float asc[4], abl[4];
    #pragma unroll
    for (int c_off = 0; c_off < 4; c_off++) {
        int c    = c_start + c_off;
        float w  = gn_w[c];
        float b  = gn_b[c];
        float sc = scale[c];
        asc[c_off] = w * inv_std * sc;
        abl[c_off] = (b - w * mean * inv_std) * sc;
    }

    const int pooled_pixels = pH * pW;
    // Each thread handles one or more pooled pixels, iterating over all 4 channels.
    for (int p = threadIdx.x; p < pooled_pixels; p += blockDim.x) {
        int ph = p / pW;
        int pw = p % pW;
        int h0 = ph * 2;
        int w0 = pw * 2;

        // Preload 2x2 raw values for each of the 4 channels.
        // Unroll channel and 2x2 window.
        #pragma unroll
        for (int c_off = 0; c_off < 4; c_off++) {
            int c = c_start + c_off;
            float max_val = -FLT_MAX;
            #pragma unroll
            for (int kh = 0; kh < 2; kh++) {
                int h_in = h0 + kh;
                if (h_in >= H) continue;
                #pragma unroll
                for (int kw = 0; kw < 2; kw++) {
                    int w_in = w0 + kw;
                    if (w_in >= W) continue;
                    float raw    = input[((n * C + c) * H + h_in) * W + w_in];
                    float normed = raw * asc[c_off] + abl[c_off];
                    if (normed > max_val) max_val = normed;
                }
            }
            max_val = fmaxf(clamp_min, fminf(clamp_max, max_val));
            output[((n * C + c) * pH + ph) * pW + pw] = max_val;
        }
    }
}

torch::Tensor fused_post_ops(
    torch::Tensor x,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor scale,
    int64_t num_groups,
    double eps,
    int64_t pool_k,
    double clamp_min,
    double clamp_max
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    x = x.to(torch::kFloat32);

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int pH = H / (int)pool_k;
    const int pW = W / (int)pool_k;

    auto gn_w = gn_weight.contiguous().to(torch::kFloat32);
    auto gn_b = gn_bias.contiguous().to(torch::kFloat32);
    auto sc   = scale.contiguous().to(torch::kFloat32).view({C});

    auto output = torch::empty({N, C, pH, pW}, x.options());

    dim3 grid(N, (int)num_groups);
    const int cpg = C / (int)num_groups;

    // Fast path: cpg==4, pool_k==2 -> specialized kernel with warp shuffles
    if (cpg == 4 && pool_k == 2) {
        const int threads = 128;
        // smem: 2 arrays of 4 floats for warp merge
        fused_gn_scale_pool_clamp_k4_p2<<<grid, threads>>>(
            x.data_ptr<float>(),
            gn_w.data_ptr<float>(),
            gn_b.data_ptr<float>(),
            sc.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W,
            (int)num_groups, (float)eps,
            pH, pW,
            (float)clamp_min, (float)clamp_max
        );
    } else {
        const int threads = 256;
        size_t smem_bytes = 2 * threads * sizeof(float);
        fused_gn_scale_pool_clamp_kernel<<<grid, threads, smem_bytes>>>(
            x.data_ptr<float>(),
            gn_w.data_ptr<float>(),
            gn_b.data_ptr<float>(),
            sc.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W,
            (int)num_groups, (float)eps,
            (int)pool_k,
            pH, pW,
            (float)clamp_min, (float)clamp_max
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs convolution, group normalization, scaling, max pooling, and clamping.
        """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x: Input tensor of shape (batch_size, in_channels, height, width).
                Returns:
                    Output tensor of shape (batch_size, out_channels, height', width').
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        pool_k = self.maxpool.kernel_size
        if isinstance(pool_k, (tuple, list)):
            pool_k = pool_k[0]
        x = _stark_get_extension().fused_post_ops(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.scale,
            self.group_norm.num_groups,
            self.group_norm.eps,
            int(pool_k),
            float(self.clamp_min),
            float(self.clamp_max),
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # scale multiply fused into fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # maxpool fused into fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # clamp fused into fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
