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
    return f'stark_cuda_l2_p19_{digest}'

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

torch::Tensor gelu_groupnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_groupnorm_cuda", &gelu_groupnorm_cuda,
          "Fused GELU + GroupNorm (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_f(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

// Vectorized two-pass fused GELU + GroupNorm kernel.
// Grid: (N * num_groups) blocks, each block handles one (n, g) group.
// Uses float4 loads/stores when data is 16-byte aligned and group_size divisible by 4.

__global__ void gelu_groupnorm_kernel_vec4(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int HW,
    int num_groups, float eps
) {
    int ng = blockIdx.x;
    int n  = ng / num_groups;
    int g  = ng % num_groups;
    int C_per_group = C / num_groups;
    int group_size  = C_per_group * HW;

    int base_c = g * C_per_group;
    long long group_offset = (long long)n * C * HW + (long long)base_c * HW;
    const float* x_group   = x   + group_offset;
    float*       out_group = out + group_offset;

    int tid        = threadIdx.x;
    int block_size = blockDim.x;

    // Check if we can use float4 (alignment + divisibility)
    bool use_vec4 = (group_size % 4 == 0) &&
                    (((uintptr_t)x_group   & 15) == 0) &&
                    (((uintptr_t)out_group & 15) == 0);

    // ---- Pass 1: Welford mean/M2 over GELU(x) ----
    float t_mean = 0.0f, t_M2 = 0.0f;
    int   t_count = 0;

    if (use_vec4) {
        const float4* x4 = reinterpret_cast<const float4*>(x_group);
        int vec_count = group_size / 4;
        for (int i = tid; i < vec_count; i += block_size) {
            float4 v4 = x4[i];
            float vals[4] = {gelu_f(v4.x), gelu_f(v4.y), gelu_f(v4.z), gelu_f(v4.w)};
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                t_count++;
                float delta = vals[k] - t_mean;
                t_mean += delta / (float)t_count;
                t_M2   += delta * (vals[k] - t_mean);
            }
        }
    } else {
        for (int i = tid; i < group_size; i += block_size) {
            int c_local = i / HW;
            int hw_idx  = i % HW;
            float v     = x_group[(long long)c_local * HW + hw_idx];
            float g_val = gelu_f(v);
            t_count++;
            float delta = g_val - t_mean;
            t_mean += delta / (float)t_count;
            t_M2   += delta * (g_val - t_mean);
        }
    }

    // Block-level Welford merge via shared memory
    extern __shared__ float smem[];
    float* s_mean  = smem;
    float* s_M2    = smem + block_size;
    float* s_count = smem + 2 * block_size;

    s_mean[tid]  = t_mean;
    s_M2[tid]    = t_M2;
    s_count[tid] = (float)t_count;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float na   = s_count[tid];
            float nb   = s_count[tid + stride];
            float n_ab = na + nb;
            if (n_ab > 0.0f) {
                float ma  = s_mean[tid];
                float mb  = s_mean[tid + stride];
                float M2a = s_M2[tid];
                float M2b = s_M2[tid + stride];
                float dlt = mb - ma;
                s_mean[tid]  = (na * ma + nb * mb) / n_ab;
                s_M2[tid]    = M2a + M2b + dlt * dlt * (na * nb / n_ab);
                s_count[tid] = n_ab;
            }
        }
        __syncthreads();
    }

    float group_mean = s_mean[0];
    float group_var  = (s_count[0] > 1.0f) ? (s_M2[0] / s_count[0]) : 0.0f;
    float inv_std    = rsqrtf(group_var + eps);

    // ---- Pass 2: recompute GELU, normalize, affine, write ----
    if (use_vec4) {
        const float4* x4   = reinterpret_cast<const float4*>(x_group);
        float4*       out4 = reinterpret_cast<float4*>(out_group);
        int vec_count = group_size / 4;
        // Precompute per-channel weight/bias for the 4 elements of each vec4
        // Each float4 at index i spans HW elements; compute which channel each belongs to
        for (int i = tid; i < vec_count; i += block_size) {
            float4 v4 = x4[i];
            // element indices within group: 4*i .. 4*i+3
            // For contiguous layout (channel-major): c_local = elem_idx / HW
            int base_elem = i * 4;
            float4 r;
            float vals[4] = {gelu_f(v4.x), gelu_f(v4.y), gelu_f(v4.z), gelu_f(v4.w)};
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int elem_idx = base_elem + k;
                int c_local  = elem_idx / HW;
                int c_global = base_c + c_local;
                float norm   = (vals[k] - group_mean) * inv_std;
                float wk     = weight[c_global];
                float bk     = bias[c_global];
                reinterpret_cast<float*>(&r)[k] = norm * wk + bk;
            }
            out4[i] = r;
        }
    } else {
        for (int i = tid; i < group_size; i += block_size) {
            int c_local  = i / HW;
            int hw_idx   = i % HW;
            int c_global = base_c + c_local;
            float v      = x_group[(long long)c_local * HW + hw_idx];
            float g_val  = gelu_f(v);
            float norm   = (g_val - group_mean) * inv_std;
            float wk     = weight[c_global];
            float bk     = bias[c_global];
            out_group[(long long)c_local * HW + hw_idx] = norm * wk + bk;
        }
    }
}

torch::Tensor gelu_groupnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N, C, H, W]");

    int N  = x.size(0);
    int C  = x.size(1);
    int H  = x.size(2);
    int W  = x.size(3);
    int HW = H * W;
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    auto out = torch::empty_like(x);

    int C_per_group = C / (int)num_groups;
    int group_size  = C_per_group * HW;
    int block_size  = 256;
    if (block_size > group_size) block_size = ((group_size + 31) / 32) * 32;
    if (block_size < 32)   block_size = 32;
    if (block_size > 1024) block_size = 1024;

    int grid = N * (int)num_groups;
    size_t smem_bytes = 3 * block_size * sizeof(float);

    gelu_groupnorm_kernel_vec4<<<grid, block_size, smem_bytes>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, HW,
        (int)num_groups,
        (float)eps
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _fused = (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous())
        if _fused:
            x = _stark_get_extension().gelu_groupnorm_cuda(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
            )
        else:
            x = torch.nn.functional.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not _fused:
                    x = self.group_norm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
