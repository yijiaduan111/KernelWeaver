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
    return f'stark_cuda_l2_p88_{digest}'

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

torch::Tensor fused_groupnorm_swish_multiply_swish(
    torch::Tensor x,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor multiply_weight,
    int64_t num_groups,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_groupnorm_swish_multiply_swish",
          &fused_groupnorm_swish_multiply_swish,
          "Fused GroupNorm + Swish + Multiply + Swish (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// One warp per (row, group) pair.
// Assumes channels_per_group == 32 (i.e. num_groups==256, out_features==8192).
// Each of the 32 lanes owns exactly one channel within its group.
// Operates in-place: out pointer == x pointer is safe because each lane
// reads its value before any lane writes, and warp shuffle keeps all
// values live in registers throughout.
__global__ void fused_gn_swish_mul_swish_warp32_kernel(
    const float* __restrict__ x,          // [rows, channels]  (read-only view)
    const float* __restrict__ gn_weight,   // [channels]
    const float* __restrict__ gn_bias,     // [channels]
    const float* __restrict__ mul_weight,  // [channels]  (broadcast over rows)
    float*       __restrict__ out,         // [rows, channels]  (may alias x)
    int rows,
    int channels,
    int num_groups,
    float eps)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;

    int total_warps = rows * num_groups;
    if (warp_id >= total_warps) return;

    int row   = warp_id / num_groups;
    int group = warp_id % num_groups;

    int channels_per_group = channels / num_groups;
    int ch = group * channels_per_group + lane;

    // Read entire group into registers before any write
    float val = x[row * channels + ch];

    // --- warp-level mean ---
    float sum = val;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    float mean = __shfl_sync(0xffffffff, sum, 0) / (float)channels_per_group;

    // --- warp-level variance ---
    float diff = val - mean;
    float var  = diff * diff;
    var += __shfl_down_sync(0xffffffff, var, 16);
    var += __shfl_down_sync(0xffffffff, var, 8);
    var += __shfl_down_sync(0xffffffff, var, 4);
    var += __shfl_down_sync(0xffffffff, var, 2);
    var += __shfl_down_sync(0xffffffff, var, 1);
    var = __shfl_sync(0xffffffff, var, 0) / (float)channels_per_group;

    float inv_std = rsqrtf(var + eps);

    // --- normalize + affine ---
    float gn_w  = gn_weight[ch];
    float gn_b  = gn_bias[ch];
    float normed = (val - mean) * inv_std * gn_w + gn_b;

    // --- first Swish ---
    float s1 = normed / (1.0f + __expf(-normed));

    // --- multiply ---
    float mw = mul_weight[ch];
    float after_mul = s1 * mw;

    // --- second Swish ---
    float result = after_mul / (1.0f + __expf(-after_mul));

    // Write in-place (all reads finished; warp is synchronised via shuffles)
    out[row * channels + ch] = result;
}

// General kernel using shared memory for arbitrary channels_per_group.
// Also operates in-place: statistics are fully computed before writes.
__global__ void fused_gn_swish_mul_swish_general_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    const float* __restrict__ mul_weight,
    float*       __restrict__ out,
    int rows,
    int channels,
    int num_groups,
    int channels_per_group,
    float eps)
{
    extern __shared__ float smem[];
    float* s_sum  = smem;
    float* s_sum2 = smem + blockDim.x;

    int group   = blockIdx.x % num_groups;
    int row     = blockIdx.x / num_groups;
    int ch_base = group * channels_per_group;
    int tid     = threadIdx.x;

    float local_sum  = 0.f;
    float local_sum2 = 0.f;
    for (int i = tid; i < channels_per_group; i += blockDim.x) {
        float v = x[row * channels + ch_base + i];
        local_sum  += v;
        local_sum2 += v * v;
    }
    s_sum[tid]  = local_sum;
    s_sum2[tid] = local_sum2;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]  += s_sum[tid + stride];
            s_sum2[tid] += s_sum2[tid + stride];
        }
        __syncthreads();
    }

    float mean    = s_sum[0]  / (float)channels_per_group;
    float var     = s_sum2[0] / (float)channels_per_group - mean * mean;
    float inv_std = rsqrtf(var + eps);
    // Statistics fully computed; safe to write in-place now

    for (int i = tid; i < channels_per_group; i += blockDim.x) {
        int ch = ch_base + i;
        float v = x[row * channels + ch];
        float normed = (v - mean) * inv_std * gn_weight[ch] + gn_bias[ch];
        float s1 = normed / (1.0f + __expf(-normed));
        float am = s1 * mul_weight[ch];
        float res = am / (1.0f + __expf(-am));
        out[row * channels + ch] = res;
    }
}

torch::Tensor fused_groupnorm_swish_multiply_swish(
    torch::Tensor x,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor multiply_weight,
    int64_t num_groups,
    double eps)
{
    TORCH_CHECK(x.is_cuda(),       "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");

    int rows     = x.size(0);
    int channels = x.size(1);
    int cpg      = channels / (int)num_groups;

    // Operate in-place: pass x's data pointer as both input and output.
    float* x_ptr = x.data_ptr<float>();

    if (cpg == 32) {
        int total_warps       = rows * (int)num_groups;
        int threads_per_block = 256; // 8 warps per block
        int blocks = (total_warps + (threads_per_block / 32) - 1) / (threads_per_block / 32);
        fused_gn_swish_mul_swish_warp32_kernel<<<blocks, threads_per_block>>>(
            x_ptr,
            gn_weight.data_ptr<float>(),
            gn_bias.data_ptr<float>(),
            multiply_weight.data_ptr<float>(),
            x_ptr,   // in-place
            rows, channels, (int)num_groups, (float)eps);
    } else {
        int threads = 128;
        int blocks  = rows * (int)num_groups;
        size_t smem = 2 * threads * sizeof(float);
        fused_gn_swish_mul_swish_general_kernel<<<blocks, threads, smem>>>(
            x_ptr,
            gn_weight.data_ptr<float>(),
            gn_bias.data_ptr<float>(),
            multiply_weight.data_ptr<float>(),
            x_ptr,   // in-place
            rows, channels, (int)num_groups, cpg, (float)eps);
    }

    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
        """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_groupnorm_swish_multiply_swish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.group_norm.num_groups,
            self.group_norm.eps
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # fused into extension above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # fused into extension above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # fused into extension above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
