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
    return f'stark_cuda_l2_p62_{digest}'

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

torch::Tensor fused_groupnorm_leakyrelu_sum_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps,
    double negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_groupnorm_leakyrelu_sum", &fused_groupnorm_leakyrelu_sum_cuda,
          "Fused GroupNorm + LeakyReLU + x+x (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda_runtime.h>

// Exact-shape specialized kernel for hidden_size=8192, num_groups=512, group_size=16.
// Uses compile-time-constant bit arithmetic to eliminate integer division/modulo.
// Two warps per block, each warp handles 2 groups (4 groups total per CTA).
__launch_bounds__(64, 4)
__global__ void fused_gn_lrelu_sum_g16_exact_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int total_groups,
    float eps,
    float negative_slope)
{
    // warp_id in [0,1], lane in [0,31], half in [0,1], local_lane in [0,15]
    const int warp_id    = threadIdx.x >> 5;
    const int lane       = threadIdx.x & 31;
    const int half       = lane >> 4;
    const int local_lane = lane & 15;

    // pair_idx selects which pair-of-groups this warp owns
    const int pair_idx  = (blockIdx.x << 1) + warp_id;
    const int group_idx = (pair_idx << 1) + half;

    const unsigned int mask = (half == 0) ? 0x0000FFFFu : 0xFFFF0000u;

    const bool valid = (group_idx < total_groups);

    float val  = 0.0f;
    int   base = 0;
    int   ch   = 0;

    if (valid) {
        // hidden_size=8192=2^13, num_groups=512=2^9, group_size=16=2^4
        // row  = group_idx / 512 = group_idx >> 9
        // grp  = group_idx % 512 = group_idx & 511
        // base = row * 8192 + grp * 16
        //      = (group_idx >> 9) * 8192 + (group_idx & 511) * 16
        //      = (group_idx >> 9) << 13 + (group_idx & 511) << 4
        //      = group_idx << 4   (since (row<<9)<<4 == row<<13 and grp<<4 == grp*16, total == group_idx<<4)
        base = group_idx << 4;  // == group_idx * 16
        // ch = grp * 16 + local_lane = (group_idx & 511) * 16 + local_lane
        ch   = ((group_idx & 511) << 4) + local_lane;
        val  = input[base + local_lane];
    }

    // One-pass reduction: accumulate sum and sumsq together
    float s  = val;
    float ss = val * val;

    s  += __shfl_down_sync(mask, s,  8);
    ss += __shfl_down_sync(mask, ss, 8);
    s  += __shfl_down_sync(mask, s,  4);
    ss += __shfl_down_sync(mask, ss, 4);
    s  += __shfl_down_sync(mask, s,  2);
    ss += __shfl_down_sync(mask, ss, 2);
    s  += __shfl_down_sync(mask, s,  1);
    ss += __shfl_down_sync(mask, ss, 1);

    const int root_lane = half << 4;
    const float mean    = __shfl_sync(mask, s,  root_lane) * (1.0f / 16.0f);
    const float sumsq   = __shfl_sync(mask, ss, root_lane);
    const float var     = fmaxf(sumsq * (1.0f / 16.0f) - mean * mean, 0.0f);
    const float inv_std = rsqrtf(var + eps);

    if (valid) {
        float norm_val = (val - mean) * inv_std;
        float out_val  = norm_val * gn_weight[ch] + gn_bias[ch];
        out_val = (out_val >= 0.0f) ? out_val : out_val * negative_slope;
        output[base + local_lane] = out_val * 2.0f;
    }
}

// Two-warps-per-block specialized kernel for group_size == 16 (generic shapes).
__launch_bounds__(64, 4)
__global__ void fused_gn_lrelu_sum_g16_2warp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int hidden_size,
    int num_groups,
    int total_groups,
    float eps,
    float negative_slope)
{
    const int warp_id    = threadIdx.x >> 5;
    const int lane       = threadIdx.x & 31;
    const int half       = lane >> 4;
    const int local_lane = lane & 15;

    const int pair_idx  = (blockIdx.x << 1) + warp_id;
    const int group_idx = (pair_idx << 1) + half;

    const unsigned int mask = (half == 0) ? 0x0000FFFFu : 0xFFFF0000u;

    const bool valid = (group_idx < total_groups);

    float val  = 0.0f;
    int   row  = 0;
    int   grp  = 0;
    int   base = 0;
    int   ch   = 0;

    if (valid) {
        row  = group_idx / num_groups;
        grp  = group_idx % num_groups;
        base = row * hidden_size + grp * 16;
        ch   = grp * 16 + local_lane;
        val  = input[base + local_lane];
    }

    float s  = val;
    float ss = val * val;

    s  += __shfl_down_sync(mask, s,  8);
    ss += __shfl_down_sync(mask, ss, 8);
    s  += __shfl_down_sync(mask, s,  4);
    ss += __shfl_down_sync(mask, ss, 4);
    s  += __shfl_down_sync(mask, s,  2);
    ss += __shfl_down_sync(mask, ss, 2);
    s  += __shfl_down_sync(mask, s,  1);
    ss += __shfl_down_sync(mask, ss, 1);

    const int root_lane = half << 4;
    const float mean    = __shfl_sync(mask, s,  root_lane) * (1.0f / 16.0f);
    const float sumsq   = __shfl_sync(mask, ss, root_lane);
    const float var     = fmaxf(sumsq * (1.0f / 16.0f) - mean * mean, 0.0f);
    const float inv_std = rsqrtf(var + eps);

    if (valid) {
        float norm_val = (val - mean) * inv_std;
        float out_val  = norm_val * gn_weight[ch] + gn_bias[ch];
        out_val = (out_val >= 0.0f) ? out_val : out_val * negative_slope;
        output[base + local_lane] = out_val * 2.0f;
    }
}

// Generic kernel for arbitrary group_size, using shared memory reduction
__global__ void fused_gn_lrelu_sum_generic_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int batch_size,
    int hidden_size,
    int num_groups,
    int group_size,
    float eps,
    float negative_slope)
{
    extern __shared__ float smem[];
    float* partial = smem;

    int group_idx = blockIdx.x;
    int row = group_idx / num_groups;
    int grp = group_idx % num_groups;
    int base = row * hidden_size + grp * group_size;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_sum = 0.0f;
    float local_sq  = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        float v = input[base + i];
        local_sum += v;
        local_sq  += v * v;
    }

    partial[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    float mean = partial[0] / (float)group_size;
    __syncthreads();

    partial[tid] = local_sq;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    float var = partial[0] / (float)group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    for (int i = tid; i < group_size; i += nthreads) {
        int ch = grp * group_size + i;
        float val = input[base + i];
        float norm_val = (val - mean) * inv_std;
        float out_val  = norm_val * gn_weight[ch] + gn_bias[ch];
        out_val = (out_val >= 0.0f) ? out_val : out_val * negative_slope;
        output[base + i] = out_val * 2.0f;
    }
}

torch::Tensor fused_groupnorm_leakyrelu_sum_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps,
    double negative_slope)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "fused_groupnorm_leakyrelu_sum: expected CUDA float32 contiguous input");
    TORCH_CHECK(x.dim() == 2, "fused_groupnorm_leakyrelu_sum: expected 2D input");

    int batch_size  = x.size(0);
    int hidden_size = x.size(1);
    int group_size  = hidden_size / (int)num_groups;

    auto output = torch::empty_like(x);

    const float* x_ptr   = x.data_ptr<float>();
    const float* w_ptr   = weight.contiguous().data_ptr<float>();
    const float* b_ptr   = bias.contiguous().data_ptr<float>();
    float*       out_ptr = output.data_ptr<float>();

    int total_groups = batch_size * (int)num_groups;

    if (group_size == 16 && hidden_size == 8192 && (int)num_groups == 512) {
        // Exact-shape fast path: all index arithmetic uses bit ops only
        int num_blocks = (total_groups + 3) >> 2;
        fused_gn_lrelu_sum_g16_exact_kernel<<<num_blocks, 64>>>(
            x_ptr, w_ptr, b_ptr, out_ptr,
            total_groups,
            (float)eps, (float)negative_slope);
    } else if (group_size == 16) {
        int num_blocks = (total_groups + 3) >> 2;
        fused_gn_lrelu_sum_g16_2warp_kernel<<<num_blocks, 64>>>(
            x_ptr, w_ptr, b_ptr, out_ptr,
            hidden_size, (int)num_groups, total_groups,
            (float)eps, (float)negative_slope);
    } else {
        int nthreads = std::min(group_size, 256);
        int nt = 1;
        while (nt < nthreads) nt <<= 1;
        nthreads = nt;
        size_t smem_bytes = nthreads * sizeof(float);
        fused_gn_lrelu_sum_generic_kernel<<<total_groups, nthreads, smem_bytes>>>(
            x_ptr, w_ptr, b_ptr, out_ptr,
            batch_size, hidden_size, (int)num_groups, group_size,
            (float)eps, (float)negative_slope);
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
        """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the forward pass of the model.

                Args:
                    x: Input tensor of shape (batch_size, input_size).

                Returns:
                    Output tensor of shape (batch_size, hidden_size).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.fc(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if x.is_cuda and x.is_contiguous() and x.dtype == torch.float32 and x.dim() == 2:
            x = _stark_get_extension().fused_groupnorm_leakyrelu_sum(
            x, self.gn.weight, self.gn.bias,
            self.gn.num_groups, self.gn.eps,
            self.leaky_relu.negative_slope)
            return x
        else:
            x = self.gn(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = self.leaky_relu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = x + x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
