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
    return f'stark_cuda_l2_p92_{digest}'

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

torch::Tensor fused_groupnorm_tail(
    torch::Tensor x_conv,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t groups,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_groupnorm_tail", &fused_groupnorm_tail,
          "Fused GroupNorm + tanh + hardswish + residual + logsumexp (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Phase-1: Compute per-(n,g) mean and invstd using Welford online algorithm.
// Grid: (N * groups), Block: up to 1024 threads covering C_per_group * H * W elements.
__global__ void groupnorm_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean_out,
    float* __restrict__ invstd_out,
    int N, int C, int H, int W,
    int groups,
    float eps)
{
    // Each block handles one (n, g) pair.
    int ng = blockIdx.x;
    int n = ng / groups;
    int g = ng % groups;
    int C_per_group = C / groups;
    int spatial = H * W;
    int group_size = C_per_group * spatial;  // elements per (n,g)

    // Base pointer for this group's data in x (NCHW layout).
    // x[n, c, h, w] = x[n*C*H*W + c*H*W + hw]
    // For group g, channels are [g*C_per_group, (g+1)*C_per_group).
    const float* x_ng = x + n * C * spatial + g * C_per_group * spatial;

    // Welford accumulation in registers with grid-stride loop.
    float welford_mean = 0.0f;
    float welford_m2 = 0.0f;
    float welford_count = 0.0f;

    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        // idx maps to (local_c, hw): local_c = idx / spatial, hw = idx % spatial.
        int local_c = idx / spatial;
        int hw = idx % spatial;
        float val = __ldg(x_ng + local_c * spatial + hw);
        // Online Welford update.
        welford_count += 1.0f;
        float delta = val - welford_mean;
        welford_mean += delta / welford_count;
        float delta2 = val - welford_mean;
        welford_m2 += delta * delta2;
    }

    // Parallel reduction across threads using shared memory.
    extern __shared__ float sdata[];  // 3 * blockDim.x floats: mean, m2, count
    float* s_mean  = sdata;
    float* s_m2    = sdata + blockDim.x;
    float* s_count = sdata + 2 * blockDim.x;

    s_mean[threadIdx.x]  = welford_mean;
    s_m2[threadIdx.x]    = welford_m2;
    s_count[threadIdx.x] = welford_count;
    __syncthreads();

    // Tree reduction to combine Welford accumulators.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a_mean  = s_mean[threadIdx.x];
            float a_m2    = s_m2[threadIdx.x];
            float a_count = s_count[threadIdx.x];
            float b_mean  = s_mean[threadIdx.x + stride];
            float b_m2    = s_m2[threadIdx.x + stride];
            float b_count = s_count[threadIdx.x + stride];
            float total   = a_count + b_count;
            if (total > 0.0f) {
                float delta   = b_mean - a_mean;
                float new_mean = a_mean + delta * (b_count / total);
                float new_m2  = a_m2 + b_m2 + delta * delta * (a_count * b_count / total);
                s_mean[threadIdx.x]  = new_mean;
                s_m2[threadIdx.x]    = new_m2;
                s_count[threadIdx.x] = total;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float final_mean   = s_mean[0];
        float final_var    = (s_count[0] > 0.0f) ? (s_m2[0] / s_count[0]) : 0.0f;
        float final_invstd = rsqrtf(final_var + eps);
        mean_out[ng]   = final_mean;
        invstd_out[ng] = final_invstd;
    }
}

// Phase-2: Per spatial location (n,h,w), apply GroupNorm affine + tanh + hardswish
// + residual add + stable logsumexp across C channels.
// Grid: (N * H * W), Block: min(C, 256) â each block handles one (n,h,w) spatial site.
__global__ void fused_tail_kernel(
    const float* __restrict__ x_conv,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int groups)
{
    int nhw = blockIdx.x;
    int n   = nhw / (H * W);
    int hw  = nhw % (H * W);
    int C_per_group = C / groups;

    // Each thread handles one or more channels.
    // Use shared memory for the max-then-sum logsumexp reduction.
    extern __shared__ float svals[];  // blockDim.x floats

    // Step 1: Each thread computes its channel value and finds local max.
    float local_max = -1e38f;
    // We'll accumulate values in svals for the second pass.
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int g = c / C_per_group;
        int ng_idx = n * groups + g;
        float x_val = __ldg(x_conv + n * C * H * W + c * H * W + hw);
        // GroupNorm affine.
        float gn_out = (x_val - mean[ng_idx]) * invstd[ng_idx] * __ldg(weight + c) + __ldg(bias + c);
        // tanh.
        float t = tanhf(gn_out);
        // hardswish: t * clamp(t+3, 0, 6) / 6.
        float hs = t * fminf(fmaxf(t + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
        // residual add with original x_conv.
        float res = x_val + hs;
        svals[threadIdx.x] = res;  // only valid for first channel per thread if C > blockDim.x
        if (res > local_max) local_max = res;
    }

    // Reduce local max across threads.
    // Use shared memory reduction.
    // We need a separate smem region for max reduction.
    // Reuse svals for max reduction first.
    // Store per-thread local_max.
    svals[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (svals[threadIdx.x + stride] > svals[threadIdx.x])
                svals[threadIdx.x] = svals[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float global_max = svals[0];
    __syncthreads();

    // Step 2: Accumulate sum of exp(val - global_max).
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int g = c / C_per_group;
        int ng_idx = n * groups + g;
        float x_val = __ldg(x_conv + n * C * H * W + c * H * W + hw);
        float gn_out = (x_val - mean[ng_idx]) * invstd[ng_idx] * __ldg(weight + c) + __ldg(bias + c);
        float t  = tanhf(gn_out);
        float hs = t * fminf(fmaxf(t + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
        float res = x_val + hs;
        local_sum += expf(res - global_max);
    }

    // Reduce sum across threads.
    svals[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            svals[threadIdx.x] += svals[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[n * H * W + hw] = global_max + logf(svals[0]);
    }
}

torch::Tensor fused_groupnorm_tail(
    torch::Tensor x_conv,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t groups,
    double eps)
{
    TORCH_CHECK(x_conv.is_cuda(), "x_conv must be a CUDA tensor");
    TORCH_CHECK(x_conv.is_contiguous(), "x_conv must be contiguous");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "weight/bias must be CUDA tensors");

    int N = x_conv.size(0);
    int C = x_conv.size(1);
    int H = x_conv.size(2);
    int W = x_conv.size(3);
    int G = (int)groups;

    TORCH_CHECK(C % G == 0, "C must be divisible by groups");

    auto opts = x_conv.options();
    auto mean_buf   = torch::empty({N * G}, opts);
    auto invstd_buf = torch::empty({N * G}, opts);

    // Phase 1: compute stats.
    int group_size = (C / G) * H * W;
    int block1 = min(group_size, 1024);
    // Round block1 down to nearest power of two for tree reduction.
    int b1 = 1;
    while (b1 * 2 <= block1) b1 *= 2;
    block1 = b1;

    int grid1 = N * G;
    size_t smem1 = 3 * block1 * sizeof(float);
    groupnorm_stats_kernel<<<grid1, block1, smem1>>>(
        x_conv.data_ptr<float>(),
        mean_buf.data_ptr<float>(),
        invstd_buf.data_ptr<float>(),
        N, C, H, W, G, (float)eps);

    // Phase 2: fused tail -> output shape (N, 1, H, W).
    auto output = torch::empty({N, 1, H, W}, opts);

    int spatial = H * W;
    int grid2 = N * spatial;
    int block2 = min(C, 256);
    // Round block2 down to power of two.
    int b2 = 1;
    while (b2 * 2 <= block2) b2 *= 2;
    block2 = b2;

    size_t smem2 = block2 * sizeof(float);
    fused_tail_kernel<<<grid2, block2, smem2>>>(
        x_conv.data_ptr<float>(),
        mean_buf.data_ptr<float>(),
        invstd_buf.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, G);

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
        Residual Addition, and LogSumExp.
        """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x_conv = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x_logsumexp = _stark_get_extension().fused_groupnorm_tail(
            x_conv.contiguous(),
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            self.group_norm.num_groups,
            float(self.group_norm.eps)
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x_logsumexp
        # <<<END_IMPROVE>>>
