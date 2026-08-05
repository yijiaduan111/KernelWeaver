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
    return f'stark_cuda_l2_p11_{digest}'

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

torch::Tensor fused_maxpool_groupnorm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_maxpool_groupnorm", &fused_maxpool_groupnorm,
          "Fused 2x2 max pool + group norm (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Kernel1: 2x2 stride-2 max pooling (unchanged)
__global__ void maxpool2x2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int Ho, int Wo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Ho * Wo;
    if (idx >= total) return;

    int wo = idx % Wo;
    int ho = (idx / Wo) % Ho;
    int c  = (idx / (Wo * Ho)) % C;
    int n  = idx / (Wo * Ho * C);

    int hi = ho * 2;
    int wi = wo * 2;

    float v = -FLT_MAX;
    #pragma unroll
    for (int dh = 0; dh < 2; dh++) {
        #pragma unroll
        for (int dw = 0; dw < 2; dw++) {
            int ih = hi + dh;
            int iw = wi + dw;
            if (ih < H && iw < W) {
                float val = input[((n * C + c) * H + ih) * W + iw];
                if (val > v) v = val;
            }
        }
    }
    output[idx] = v;
}

// Kernel 2: GroupNorm over pooled output with reduced register pressure.
// Fixed 128-thread launch with __launch_bounds__ to lower register allocation.
__global__ __launch_bounds__(128, 2)
void groupnorm_kernel(
    const float* __restrict__ pooled,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int Ho, int Wo,
    int num_groups, float eps)
{
    int ng      = blockIdx.x;
    int n       = ng / num_groups;
    int g       = ng % num_groups;
    int cpg     = C / num_groups;
    int spatial = Ho * Wo;
    int group_size = cpg * spatial;
    int c_start = g * cpg;

    // --- Pass 1: accumulate partial sum and sum-of-squares ---
    float local_sum  = 0.f;
    float local_sum2 = 0.f;

    for (int i = threadIdx.x; i < group_size; i += 128) {
        int lc = i / spatial;
        int s  = i % spatial;
        int ho = s / Wo;
        int wo = s % Wo;
        float v = pooled[((n * C + c_start + lc) * Ho + ho) * Wo + wo];
        local_sum  += v;
        local_sum2 += v * v;
    }

    // --- Warp-level reduction ---
    const unsigned FULL_MASK = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum  += __shfl_down_sync(FULL_MASK, local_sum,  offset);
        local_sum2 += __shfl_down_sync(FULL_MASK, local_sum2, offset);
    }

    // --- Cross-warp combine via small shared buffer ---
    int lane_id  = threadIdx.x & 31;
    int warp_id  = threadIdx.x >> 5;
    const int num_warps = 4;  // 128 threads = 4 warps

    __shared__ float warp_sum[4];
    __shared__ float warp_sum2[4];

    if (lane_id == 0) {
        warp_sum[warp_id]  = local_sum;
        warp_sum2[warp_id] = local_sum2;
    }
    __syncthreads();

    // First warp reduces the per-warp partial sums
    float total_sum  = (threadIdx.x < num_warps) ? warp_sum[threadIdx.x]  : 0.f;
    float total_sum2 = (threadIdx.x < num_warps) ? warp_sum2[threadIdx.x] : 0.f;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            total_sum  += __shfl_down_sync(FULL_MASK, total_sum,  offset);
            total_sum2 += __shfl_down_sync(FULL_MASK, total_sum2, offset);
        }
    }

    // Broadcast mean and inv_std via two static shared scalars
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float mean    = total_sum / (float)group_size;
        float var     = total_sum2 / (float)group_size - mean * mean;
        s_mean        = mean;
        s_inv_std     = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean    = s_mean;
    float inv_std = s_inv_std;

    // --- Pass 2: apply affine and write output ---
    for (int i = threadIdx.x; i < group_size; i += 128) {
        int lc = i / spatial;
        int s  = i % spatial;
        int ho = s / Wo;
        int wo = s % Wo;
        int c  = c_start + lc;
        int out_idx = ((n * C + c) * Ho + ho) * Wo + wo;
        float v = pooled[out_idx];
        output[out_idx] = (v - mean) * inv_std * __ldg(&weight[c]) + __ldg(&bias[c]);
    }
}

torch::Tensor fused_maxpool_groupnorm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Ho = H / 2;
    int Wo = W / 2;

    auto pooled = torch::empty({N, C, Ho, Wo}, x.options());

    // Launch max pool kernel
    int total_pool = N * C * Ho * Wo;
    int threads_pool = 256;
    int blocks_pool = (total_pool + threads_pool - 1) / threads_pool;
    maxpool2x2_kernel<<<blocks_pool, threads_pool>>>(
        x.data_ptr<float>(),
        pooled.data_ptr<float>(),
        N, C, H, W, Ho, Wo);

    auto out = torch::empty({N, C, Ho, Wo}, x.options());

    int num_groups_i = (int)num_groups;
    int num_blocks = N * num_groups_i;

    // Fixed 128-thread launch to reduce register pressure
    groupnorm_kernel<<<num_blocks, 128>>>(
        pooled.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, Ho, Wo,
        num_groups_i, (float)eps);

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.batch_norm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.tanh(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if x.is_cuda and x.dtype == torch.float32:
            x = _stark_get_extension().fused_maxpool_groupnorm(
                x.contiguous(),
                self.group_norm.weight.contiguous(),
                self.group_norm.bias.contiguous(),
                self.group_norm.num_groups,
                self.group_norm.eps)
            _fused_pool_gn = True
        else:
            x = self.max_pool(x)
            _fused_pool_gn = False
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if not _fused_pool_gn:
            x = self.group_norm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
