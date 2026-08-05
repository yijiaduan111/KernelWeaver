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
    return f'stark_cuda_l2_p94_{digest}'

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

torch::Tensor gemm_biasadd_hardtanh_mish_groupnorm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor extra_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_biasadd_hardtanh_mish_groupnorm",
          &gemm_biasadd_hardtanh_mish_groupnorm,
          "Fused GEMM epilogue: BiasAdd + Hardtanh + Mish + GroupNorm (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

// Each block handles one (batch_row, group) pair.
// blockDim.x must equal group_size (32 for the default config).
__global__ void fused_epilogue_kernel(
    const float* __restrict__ gemm_out,   // [batch, out_features]
    const float* __restrict__ extra_bias, // [out_features]
    const float* __restrict__ gn_weight,  // [out_features]
    const float* __restrict__ gn_bias,    // [out_features]
    float* __restrict__ output,           // [batch, out_features]
    int out_features,
    int group_size,
    int num_groups,
    float eps
) {
    const int row   = blockIdx.x;  // batch index
    const int grp   = blockIdx.y;  // group index
    const int lane  = threadIdx.x; // lane within group [0, group_size)

    const int col = grp * group_size + lane;

    // Load GEMM output, add extra bias
    float v = gemm_out[row * out_features + col] + extra_bias[col];

    // Hardtanh: clamp to [-1, 1]
    v = fmaxf(-1.0f, fminf(1.0f, v));

    // Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    // Numerically stable form:
    // softplus(x) = x + log(1 + exp(-x))  for x >= 0
    //             = log(1 + exp(x))        for x < 0
    float sp;
    if (v >= 0.0f) {
        sp = v + log1pf(expf(-v));
    } else {
        sp = log1pf(expf(v));
    }
    v = v * tanhf(sp);

    // Warp reduction for mean and variance (group_size == warp size == 32)
    float sum = v;
    float sum_sq = v * v;

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        sum    += __shfl_down_sync(0xffffffff, sum,    offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Broadcast mean and variance from lane 0
    float mean = __shfl_sync(0xffffffff, sum,    0) / (float)group_size;
    float var  = __shfl_sync(0xffffffff, sum_sq, 0) / (float)group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine parameters
    float normalized = (v - mean) * inv_std;
    float out_val = normalized * gn_weight[col] + gn_bias[col];

    output[row * out_features + col] = out_val;
}

torch::Tensor gemm_biasadd_hardtanh_mish_groupnorm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor extra_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    // GEMM: x @ weight.T + linear_bias  (ATen linear)
    auto gemm_out = at::linear(x, weight, linear_bias);
    gemm_out = gemm_out.contiguous();

    const int batch_size   = x.size(0);
    const int out_features = (int)weight.size(0);
    const int group_size   = out_features / (int)num_groups;

    TORCH_CHECK(group_size == 32, "group_size must be 32 for this kernel");

    auto output = torch::empty_like(gemm_out);

    // Grid: (batch_size, num_groups), Block: (group_size,)
    // One warp per (batch_row, group) pair; tuned for group_size == 32.
    dim3 grid(batch_size, (int)num_groups);
    dim3 block(group_size);

    fused_epilogue_kernel<<<grid, block>>>(
        gemm_out.data_ptr<float>(),
        extra_bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        out_features,
        group_size,
        (int)num_groups,
        (float)eps
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
        """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        group_size = self.gemm.out_features // self.groupnorm.num_groups
        if (
            x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and
            self.gemm.weight.is_cuda and self.gemm.weight.dtype == torch.float32 and
            self.bias.is_cuda and
            self.groupnorm.weight is not None and self.groupnorm.bias is not None and
            group_size == 32
        ):
            return _stark_get_extension().gemm_biasadd_hardtanh_mish_groupnorm(
                x,
                self.gemm.weight,
                self.gemm.bias,
                self.bias,
                self.groupnorm.weight,
                self.groupnorm.bias,
                self.groupnorm.num_groups,
                self.groupnorm.eps
            )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x + self.bias
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = self.hardtanh(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = self.mish(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        x = self.groupnorm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
