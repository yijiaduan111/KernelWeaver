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
    return f'stark_cuda_l2_p58_{digest}'

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

torch::Tensor fused_logsumexp_hardswish_subtract_clamp_cuda(
    torch::Tensor x,
    torch::Tensor bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_logsumexp_hardswish_subtract_clamp_cuda",
          &fused_logsumexp_hardswish_subtract_clamp_cuda,
          "Fused logsumexp + hardswish + subtract + clamp (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Single-pass online logsumexp kernel with reduced register pressure.
// Bias is read from device memory to avoid implicit host-device sync from .item<float>().
__global__ __launch_bounds__(256, 2)
void fused_lse_hardswish_sub_clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int N,
    int C,
    int spatial_size
) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * spatial_size;
    if (spatial_idx >= total) return;

    int n = spatial_idx / spatial_size;
    int s = spatial_idx % spatial_size;

    // Input layout: NCDHW -> channel stride = spatial_size
    const float* base = input + (n * C) * spatial_size + s;

    // Single-pass online logsumexp: avoids two full channel scans and
    // halves the number of live accumulator registers.
    float max_val = base[0];
    float sum_exp = 1.0f;  // exp(base[0] - max_val) = exp(0) = 1

    for (int c = 1; c < C; c++) {
        float v = base[c * spatial_size];
        if (v > max_val) {
            // Correct running sum for new max
            sum_exp = sum_exp * __expf(max_val - v) + 1.0f;
            max_val = v;
        } else {
            sum_exp += __expf(v - max_val);
        }
    }

    float lse = max_val + __logf(sum_exp);

    // HardSwish: x * sigmoid(x+3) / 6
    float sig = 1.0f / (1.0f + __expf(-(lse + 3.0f)));
    float hs = lse * sig * (1.0f / 6.0f);

    // Subtract bias (read once from device), clamp to [-1, 1]
    float result = hs - bias[0];
    result = fminf(1.0f, fmaxf(-1.0f, result));

    output[spatial_idx] = result;
}

torch::Tensor fused_logsumexp_hardswish_subtract_clamp_cuda(
    torch::Tensor x,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int spatial_size = D * H * W;
    int total = N * spatial_size;

    auto output = torch::empty({N, 1, D, H, W}, x.options());

    // Pass bias as device pointer to avoid implicit device->host sync from .item<float>()
    const float* bias_ptr = bias.data_ptr<float>();

    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    fused_lse_hardswish_sub_clamp_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
        N, C, spatial_size
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        x = self.conv_transpose(x)
        return _stark_get_extension().fused_logsumexp_hardswish_subtract_clamp_cuda(x.contiguous(), self.bias)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = torch.logsumexp(x, dim=1, keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x * torch.sigmoid(x + 3) / 6
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = x - self.bias
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = torch.clamp(x, min=-1, max=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
