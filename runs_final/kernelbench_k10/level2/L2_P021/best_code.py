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
    return f'stark_cuda_l2_p21_{digest}'

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

torch::Tensor fused_post_conv(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused bias+scale+sigmoid+GroupNorm with float4 vectorized spatial loops");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ void welford_merge(
    float& count_a, float& mean_a, float& M2_a,
    float  count_b, float  mean_b, float  M2_b
) {
    float combined = count_a + count_b;
    if (combined == 0.0f) return;
    float delta = mean_b - mean_a;
    mean_a = (count_a * mean_a + count_b * mean_b) / combined;
    M2_a   = M2_a + M2_b + delta * delta * (count_a * count_b / combined);
    count_a = combined;
}

__device__ __forceinline__ void warp_welford_reduce(
    float& count, float& mean, float& M2
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float c_b  = __shfl_down_sync(0xffffffff, count, offset);
        float m_b  = __shfl_down_sync(0xffffffff, mean,  offset);
        float m2_b = __shfl_down_sync(0xffffffff, M2,    offset);
        int lane = threadIdx.x &31;
        if (lane < offset) {
            welford_merge(count, mean, M2, c_b, m_b, m2_b);
        }
    }
}

__device__ __forceinline__ float apply_bss(float val, float b, float s) {
    val = (val + b) * s;
    return 1.0f / (1.0f + __expf(-val));
}

__global__ __launch_bounds__(128, 4) void fused_bias_scale_sigmoid_groupnorm_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    int N, int C, int H, int W,
    int num_groups, float eps
) {
    int ng = blockIdx.x;
    int n= ng / num_groups;
    int g  = ng % num_groups;
    int C_per_group = C / num_groups;
    int spatial     = H * W;

    int base_channel = g * C_per_group;
    int base_offset  = n * C * spatial + base_channel * spatial;

    int tid= threadIdx.x;
    int block_size = blockDim.x;  // 128
    int warp_id    = tid >> 5;
    int lane_id    = tid & 31;
    int num_warps  = block_size >> 5;  // 4

    int spatial4= spatial >> 2;

    float thread_count = 0.0f;
    float thread_mean  = 0.0f;
    float thread_M2    = 0.0f;

    // Cache per-channel bias/scale in shared memory to reduce global-memory register pressure
    extern __shared__ float smem[];
    float* s_bias  = smem;   // C_per_group floats
    float* s_scale = smem + C_per_group;     // C_per_group floats

    if (tid < C_per_group) {
        s_bias [tid] = bias [base_channel + tid];
        s_scale[tid] = scale[base_channel + tid];
    }
    __syncthreads();

    #pragma unroll 1
    for (int c_local = 0; c_local < C_per_group; ++c_local) {
        float b = s_bias [c_local];
        float s = s_scale[c_local];
        const float* ch_ptr = x_in + base_offset + c_local * spatial;
        const float4* ch_ptr4 = reinterpret_cast<const float4*>(ch_ptr);

        for (int i4 = tid; i4 < spatial4; i4 += block_size) {
            float4 v = ch_ptr4[i4];
            float v0 = apply_bss(v.x, b, s);
            float v1 = apply_bss(v.y, b, s);
            float v2 = apply_bss(v.z, b, s);
            float v3 = apply_bss(v.w, b, s);
            thread_count += 1.0f; float delta = v0 - thread_mean; thread_mean += delta / thread_count; thread_M2 += delta * (v0 - thread_mean);
            thread_count += 1.0f;delta = v1 - thread_mean; thread_mean += delta / thread_count; thread_M2 += delta * (v1 - thread_mean);
            thread_count += 1.0f;       delta = v2 - thread_mean; thread_mean += delta / thread_count; thread_M2 += delta * (v2 - thread_mean);
            thread_count += 1.0f;       delta = v3 - thread_mean; thread_mean += delta / thread_count; thread_M2 += delta * (v3 - thread_mean);
        }
int rem_start = spatial4 * 4;
        for (int sp = rem_start + tid; sp < spatial; sp += block_size) {
            float val = apply_bss(ch_ptr[sp], b, s);
            thread_count += 1.0f;
            float delta = val - thread_mean;
            thread_mean += delta / thread_count;
            thread_M2   += delta * (val - thread_mean);
        }
    }

    warp_welford_reduce(thread_count, thread_mean, thread_M2);

    // Reuse smem tail for warp-reduction scratch (after bias/scale are consumed)
    float* s_warp_count = smem + 2 * C_per_group;
    float* s_warp_mean  = s_warp_count + 4;
    float* s_warp_M2    = s_warp_mean  + 4;
    float* s_finals     = s_warp_M2    + 4;  // [0]=mean, [1]=inv_std

    if (lane_id == 0) {
        s_warp_count[warp_id] = thread_count;
        s_warp_mean [warp_id] = thread_mean;
        s_warp_M2   [warp_id] = thread_M2;
    }
    __syncthreads();

    if (warp_id == 0) {
        float count = (lane_id < num_warps) ? s_warp_count[lane_id] : 0.0f;
        float mean  = (lane_id < num_warps) ? s_warp_mean [lane_id] : 0.0f;
        float M2    = (lane_id < num_warps) ? s_warp_M2   [lane_id] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float c_b  = __shfl_down_sync(0xffffffff, count, offset);
            float m_b  = __shfl_down_sync(0xffffffff, mean,  offset);
            float m2_b = __shfl_down_sync(0xffffffff, M2,    offset);
            if (lane_id < offset) {
                welford_merge(count, mean, M2, c_b, m_b, m2_b);
            }
        }

        if (lane_id == 0) {
            float group_var = M2 / (count > 0.0f ? count : 1.0f);
            s_finals[0] = mean;
            s_finals[1] = rsqrtf(group_var + eps);
        }
    }
    __syncthreads();

    float group_mean = s_finals[0];
    float inv_std    = s_finals[1];

    #pragma unroll 1
    for (int c_local = 0; c_local < C_per_group; ++c_local) {
        int c = base_channel + c_local;
        float b  = s_bias [c_local];
        float s  = s_scale[c_local];
        float gw = gn_weight[c];
        float gb = gn_bias[c];

        const float*  ch_in_f  = x_in  + base_offset + c_local * spatial;
        float*        ch_out_f = x_out + base_offset + c_local * spatial;
        const float4* ch_in4   = reinterpret_cast<const float4*>(ch_in_f);
        float4*       ch_out4  = reinterpret_cast<float4*>(ch_out_f);

        for (int i4 = tid; i4 < spatial4; i4 += block_size) {
            float4 v = ch_in4[i4];
            float4 o;
            o.x = (apply_bss(v.x, b, s) - group_mean) * inv_std * gw + gb;
            o.y = (apply_bss(v.y, b, s) - group_mean) * inv_std * gw + gb;
            o.z = (apply_bss(v.z, b, s) - group_mean) * inv_std * gw + gb;
            o.w = (apply_bss(v.w, b, s) - group_mean) * inv_std * gw + gb;
            ch_out4[i4] = o;
        }
        int rem_start = spatial4 * 4;
        for (int sp = rem_start + tid; sp < spatial; sp += block_size) {
            float val = apply_bss(ch_in_f[sp], b, s);
            ch_out_f[sp] = (val - group_mean) * inv_std * gw + gb;
        }
    }
}

torch::Tensor fused_post_conv(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto x_out = torch::empty_like(x);

    auto bias_1d  = bias.view({C});
    auto scale_1d = scale.view({C});

    int C_per_group = C / num_groups;
    int grid_size  = N * num_groups;
    int block_size = 128;
    // smem: 2*C_per_group (bias+scale) + 4*3 (warp scratch) + 2(finals)
    size_t smem_bytes = (2 * C_per_group + 14) * sizeof(float);

    fused_bias_scale_sigmoid_groupnorm_kernel<<<grid_size, block_size, smem_bytes>>>(
        x.data_ptr<float>(),
        x_out.data_ptr<float>(),
        bias_1d.data_ptr<float>(),
        scale_1d.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        N, C, H, W,
        num_groups, eps
    );

    return x_out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
        """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_post_conv(
            x,
            self.bias.view(-1),
            self.scale.view(-1),
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
