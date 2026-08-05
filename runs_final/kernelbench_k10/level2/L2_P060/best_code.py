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
    return f'stark_cuda_l2_p60_{digest}'

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

torch::Tensor swish_groupnorm_hardswish_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish_groupnorm_hardswish", &swish_groupnorm_hardswish_cuda,
          "Fused Swish + GroupNorm + HardSwish (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Specialized fast-path kernel for C=16, num_groups=4 (4 channels per group)
// Loops by channel then spatial to eliminate per-element division/modulo.
// Affine scalars for all 4 channels cached in registers before writeback pass.
__global__ void __launch_bounds__(128, 4)
swish_groupnorm_hardswish_specialized_cpg4(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int D, int H, int W,
    float eps
) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int spatial = D * H * W;

    // base points to element [n, g*4, 0, 0, 0] in NCDHW layout
    int64_t base = ((int64_t)n * 16 + (int64_t)g * 4) * spatial;

    __shared__ float warp_sums[4];
    __shared__ float warp_sumsqs[4];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Statistics pass: iterate channel-major to avoid division in index calc
    for (int c_local = 0; c_local < 4; c_local++) {
        int64_t ch_base = base + (int64_t)c_local * spatial;
        for (int s = threadIdx.x; s < spatial; s += 128) {
            float v = __ldg(x + ch_base + s);
            float sv = v * (1.0f / (1.0f + __expf(-v)));
            local_sum += sv;
            local_sum_sq += sv * sv;
        }
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum    += __shfl_down_sync(0xffffffff, local_sum,    offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    if (lane == 0) {
        warp_sums[warp_id]   = local_sum;
        warp_sumsqs[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Cross-warp reduction: 4 warps -> 4 values, reduce in first warp
    float group_sum    = 0.0f;
    float group_sum_sq = 0.0f;
    if (threadIdx.x < 4) {
        group_sum    = warp_sums[threadIdx.x];
        group_sum_sq = warp_sumsqs[threadIdx.x];
    }
    for (int offset = 2; offset > 0; offset >>= 1) {
        group_sum    += __shfl_down_sync(0xf, group_sum,    offset);
        group_sum_sq += __shfl_down_sync(0xf, group_sum_sq, offset);
    }
    if (threadIdx.x == 0) {
        warp_sums[0]   = group_sum;
        warp_sumsqs[0] = group_sum_sq;
    }
    __syncthreads();

    int elements_per_group = 4 * spatial;
    float inv_n  = 1.0f / (float)elements_per_group;
    float mean   = warp_sums[0] * inv_n;
    float var    = warp_sumsqs[0] * inv_n - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Cache the 4 affine scalars in registers
    int g4 = g * 4;
    float w0 = __ldg(weight + g4 + 0), b0 = __ldg(bias + g4 + 0);
    float w1 = __ldg(weight + g4 + 1), b1 = __ldg(bias + g4 + 1);
    float w2 = __ldg(weight + g4 + 2), b2 = __ldg(bias + g4 + 2);
    float w3 = __ldg(weight + g4 + 3), b3 = __ldg(bias + g4 + 3);

    // Writeback pass: same channel-major iteration, no division needed
    for (int c_local = 0; c_local < 4; c_local++) {
        int64_t ch_base = base + (int64_t)c_local * spatial;
        float wc, bc;
        if      (c_local == 0) { wc = w0; bc = b0; }
        else if (c_local == 1) { wc = w1; bc = b1; }
        else if (c_local == 2) { wc = w2; bc = b2; }
        else                   { wc = w3; bc = b3; }

        for (int s = threadIdx.x; s < spatial; s += 128) {
            float v  = __ldg(x + ch_base + s);
            float sv = v * (1.0f / (1.0f + __expf(-v)));
            float normed = (sv - mean) * inv_std;
            float y = normed * wc + bc;

            float hs;
            if      (y <= -3.0f) hs = 0.0f;
            else if (y >=  3.0f) hs = y;
            else                 hs = y * (y + 3.0f) * (1.0f / 6.0f);

            out[ch_base + s] = hs;
        }
    }
}

// Generic fallback kernel for arbitrary C and num_groups
__global__ void __launch_bounds__(256, 4)
swish_groupnorm_hardswish_generic(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int D, int H, int W,
    int num_groups,
    float eps
) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int channels_per_group = C / num_groups;
    int spatial = D * H * W;
    int elements_per_group = channels_per_group * spatial;

    int64_t base = (int64_t)n * C * spatial + (int64_t)g * channels_per_group * spatial;

    extern __shared__ float smem[];
    float* ssum   = smem;
    float* ssumsq = smem + blockDim.x;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < elements_per_group; i += blockDim.x) {
        int c_local = i / spatial;
        int s       = i - c_local * spatial;
        int64_t idx = base + (int64_t)c_local * spatial + s;
        float v  = __ldg(x + idx);
        float sv = v * (1.0f / (1.0f + __expf(-v)));
        local_sum    += sv;
        local_sum_sq += sv * sv;
    }

    ssum[threadIdx.x]   = local_sum;
    ssumsq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ssum[threadIdx.x]   += ssum[threadIdx.x + stride];
            ssumsq[threadIdx.x] += ssumsq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_n  = 1.0f / (float)elements_per_group;
    float mean   = ssum[0] * inv_n;
    float var    = ssumsq[0] * inv_n - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < elements_per_group; i += blockDim.x) {
        int c_local = i / spatial;
        int s       = i - c_local * spatial;
        int c       = g * channels_per_group + c_local;
        int64_t idx = base + (int64_t)c_local * spatial + s;

        float v  = __ldg(x + idx);
        float sv = v * (1.0f / (1.0f + __expf(-v)));

        float normed = (sv - mean) * inv_std;
        float y = normed * __ldg(weight + c) + __ldg(bias + c);

        float hs;
        if      (y <= -3.0f) hs = 0.0f;
        else if (y >=  3.0f) hs = y;
        else                 hs = y * (y + 3.0f) * (1.0f / 6.0f);

        out[idx] = hs;
    }
}

torch::Tensor swish_groupnorm_hardswish_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    TORCH_CHECK(x.is_cuda(),        "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),  "x must be contiguous");
    TORCH_CHECK(x.dim() == 5,       "x must be 5-dimensional (N,C,D,H,W)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int D = (int)x.size(2);
    int H = (int)x.size(3);
    int W = (int)x.size(4);

    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    auto out = torch::empty_like(x);

    dim3 grid((unsigned)N, (unsigned)num_groups);

    if (C == 16 && num_groups == 4) {
        dim3 block_spec(128);
        swish_groupnorm_hardswish_specialized_cpg4<<<grid, block_spec>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            N, D, H, W,
            (float)eps
        );
    } else {
        const int block_size = 256;
        dim3 block_gen(block_size);
        size_t smem_bytes = 2 * block_size * sizeof(float);
        swish_groupnorm_hardswish_generic<<<grid, block_gen, smem_bytes>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, D, H, W,
            (int)num_groups,
            (float)eps
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, applies Swish activation, 
        group normalization, and then HardSwish activation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().swish_groupnorm_hardswish(
                    x,
                    self.group_norm.weight,
                    self.group_norm.bias,
                    self.group_norm.num_groups,
                    float(self.group_norm.eps)
                )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # GroupNorm fused into swish_groupnorm_hardswish extension above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # HardSwish fused into swish_groupnorm_hardswish extension above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
