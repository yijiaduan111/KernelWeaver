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
    return f'stark_cuda_l2_p37_{digest}'

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

torch::Tensor fused_swish_bias_groupnorm(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_swish_bias_groupnorm", &fused_swish_bias_groupnorm,
          "Fused Swish + bias add + GroupNorm (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda_runtime.h>

// One 128-thread block processes 4 consecutive groups from the same sample.
// Each 32-lane warp owns one 64-channel group; each lane handles 2 contiguous
// channels via float2 vectorized loads/stores, so all reduction is warp-local --
// no smem, no __syncthreads.
// Affine parameters (gamma, beta) are loaded AFTER the warp reduction to
// keep fewer live registers during the reduction stage.
__global__ void __launch_bounds__(128, 2) fused_swish_bias_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C, int num_groups, float eps
) {
    const int sample  = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int group   = blockIdx.y * 4 + warp_id;

    if (group >= num_groups) return;

    // Each lane covers 2 contiguous channels within this group
    const int c0   = group * 64 + lane * 2;
    const int base = sample * C;

    // Vectorized loads for input and bias only; defer gamma/beta until after reduction
    float2 in_v   = __ldg(reinterpret_cast<const float2*>(input + base + c0));
    float2 bias_v = __ldg(reinterpret_cast<const float2*>(bias + c0));

    // Apply Swish (x * sigmoid(x)) and add bias -- keep pre0/pre1 register-resident
    float pre0 = in_v.x * (1.0f / (1.0f + __expf(-in_v.x))) + bias_v.x;
    float pre1 = in_v.y * (1.0f / (1.0f + __expf(-in_v.y))) + bias_v.y;

    // Per-lane accumulation over 2 elements
    float sum   = pre0 + pre1;
    float sumsq = pre0 * pre0 + pre1 * pre1;

    // Warp-local reduction across 32 lanes (covers all 64 channels)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum   += __shfl_down_sync(0xffffffffu, sum,   offset);
        sumsq += __shfl_down_sync(0xffffffffu, sumsq, offset);
    }

    // Compute mean/inv_std on lane 0, then broadcast to all lanes
    float mean    = 0.0f;
    float inv_std = 0.0f;
    if (lane == 0) {
        mean    = sum * (1.0f / 64.0f);
        float var = sumsq * (1.0f / 64.0f) - mean * mean;
        inv_std = rsqrtf(var + eps);
    }
    mean    = __shfl_sync(0xffffffffu, mean,    0);
    inv_std = __shfl_sync(0xffffffffu, inv_std, 0);

    // Load affine parameters now (after reduction) to minimize live registers during reduce
    float2 gam_v = __ldg(reinterpret_cast<const float2*>(gamma + c0));
    float2 bet_v = __ldg(reinterpret_cast<const float2*>(beta + c0));

    // Apply GroupNorm affine transform and write output via float2 store
    float2 out_v;
    out_v.x = gam_v.x * (pre0 - mean) * inv_std + bet_v.x;
    out_v.y = gam_v.y * (pre1 - mean) * inv_std + bet_v.y;
    *reinterpret_cast<float2*>(output + base + c0) = out_v;
}

torch::Tensor fused_swish_bias_groupnorm(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input must be float32");

    const int N        = input.size(0);
    const int C        = input.size(1);
    const int group_sz = C / num_groups;
    TORCH_CHECK(group_sz == 64,
                "This kernel specializes for group_sz == 64");
    TORCH_CHECK(num_groups % 4 == 0,
                "num_groups must be divisible by 4 for the 4-group-per-block kernel");

    auto output = torch::empty_like(input);

    const int grid_y = (num_groups + 3) / 4;
    dim3 grid(N, grid_y);
    dim3 block(128);

    fused_swish_bias_groupnorm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, num_groups, eps
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
        """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_features).
                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_features).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.matmul(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().fused_swish_bias_groupnorm(
            x.contiguous(),
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            float(self.group_norm.eps)
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
