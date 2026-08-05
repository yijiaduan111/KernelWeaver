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
    return f'stark_cuda_l2_p61_{digest}'

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

torch::Tensor relu_groupnorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups, double eps);

torch::Tensor relu_groupnorm(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N, C, D, H, W)");
    int64_t C = x.size(1);
    TORCH_CHECK(C % num_groups == 0, "channels must be divisible by num_groups");
    return relu_groupnorm_cuda(x.contiguous(), weight.contiguous(), bias.contiguous(), num_groups, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_groupnorm", &relu_groupnorm, "Fused ReLU + GroupNorm (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Two-pass fused ReLU+GroupNorm kernel with shared-memory affine caching.
// Grid: (N * num_groups) blocks, each block handles one (n, g) pair.
// Dynamic shared memory layout (floats):
//   [0 .. BLOCK_SIZE)                        : s_mean  (Welford reduction)
//   [BLOCK_SIZE .. 2*BLOCK_SIZE)             : s_m2    (Welford reduction)
//   [2*BLOCK_SIZE .. 3*BLOCK_SIZE)           : s_cnt   (Welford reduction)
//   [3*BLOCK_SIZE .. 3*BLOCK_SIZE+CPG)       : s_weight (affine scale)
//   [3*BLOCK_SIZE+CPG .. 3*BLOCK_SIZE+2*CPG) : s_bias  (affine bias)

template <int BLOCK_SIZE>
__global__ void relu_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int num_groups,
    float eps
) {
    int ng = blockIdx.x;
    int n  = ng / num_groups;
    int g  = ng % num_groups;
    int C_per_group = C / num_groups;
    int spatial = D * H * W;
    int group_elems = C_per_group * spatial;

    int base_channel = g * C_per_group;
    const float* in_ptr  = input  + (int64_t)n * C * spatial + base_channel * spatial;
    float*       out_ptr = output + (int64_t)n * C * spatial + base_channel * spatial;

    extern __shared__ float smem[];
    float* s_mean   = smem;
    float* s_m2     = smem + BLOCK_SIZE;
    float* s_cnt    = smem + 2 * BLOCK_SIZE;
    float* s_weight = smem + 3 * BLOCK_SIZE;
    float* s_bias   = smem + 3 * BLOCK_SIZE + C_per_group;

    // Stage affine parameters for this group into shared memory.
    // The first __syncthreads() in the reduction ensures visibility before pass 2.
    for (int c = threadIdx.x; c < C_per_group; c += BLOCK_SIZE) {
        s_weight[c] = weight[base_channel + c];
        s_bias[c]   = bias_ptr[base_channel + c];
    }

    // --- Pass 1: Welford online mean/variance over relu(x) ---
    float mean_acc = 0.0f;
    float m2_acc   = 0.0f;
    float count    = 0.0f;

    for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
        float val = in_ptr[i];
        float r   = val > 0.0f ? val : 0.0f;
        count += 1.0f;
        float delta = r - mean_acc;
        mean_acc += delta / count;
        float delta2 = r - mean_acc;
        m2_acc += delta * delta2;
    }

    s_mean[threadIdx.x] = mean_acc;
    s_m2[threadIdx.x]   = m2_acc;
    s_cnt[threadIdx.x]  = count;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float cnt_a  = s_cnt[threadIdx.x];
            float cnt_b  = s_cnt[threadIdx.x + stride];
            float mean_a = s_mean[threadIdx.x];
            float mean_b = s_mean[threadIdx.x + stride];
            float m2_a   = s_m2[threadIdx.x];
            float m2_b   = s_m2[threadIdx.x + stride];
            float cnt_ab  = cnt_a + cnt_b;
            float delta   = mean_b - mean_a;
            float mean_ab = (cnt_ab > 0.0f) ? (mean_a + delta * cnt_b / cnt_ab) : 0.0f;
            float m2_ab   = m2_a + m2_b + delta * delta * cnt_a * cnt_b / (cnt_ab > 0.0f ? cnt_ab : 1.0f);
            s_cnt[threadIdx.x]  = cnt_ab;
            s_mean[threadIdx.x] = mean_ab;
            s_m2[threadIdx.x]   = m2_ab;
        }
        __syncthreads();
    }

    float final_mean = s_mean[0];
    float final_var  = s_m2[0] / (float)group_elems;
    float inv_std    = rsqrtf(final_var + eps);

    // --- Pass 2: ReLU + normalize + affine (affine from shared memory cache) ---
    for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
        int c_local = i / spatial;
        float val  = in_ptr[i];
        float r    = val > 0.0f ? val : 0.0f;
        float norm = (r - final_mean) * inv_std;
        out_ptr[i] = norm * s_weight[c_local] + s_bias[c_local];
    }
}

torch::Tensor relu_groupnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    auto output = torch::empty_like(x);

    int grid = N * (int)num_groups;
    const int BLOCK = 256;
    int C_per_group = C / (int)num_groups;

    // Dynamic shared memory: 3*BLOCK floats for Welford + 2*C_per_group floats for affine cache
    size_t smem_bytes = (size_t)(3 * BLOCK + 2 * C_per_group) * sizeof(float);

    relu_groupnorm_kernel<BLOCK><<<grid, BLOCK, smem_bytes>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        (int)num_groups,
        (float)eps
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
        """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().relu_groupnorm(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups, self.group_norm.eps)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # normalization already applied by fused relu_groupnorm above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
