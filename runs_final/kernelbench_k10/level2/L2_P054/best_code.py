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
    return f'stark_cuda_l2_p54_{digest}'

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

torch::Tensor conv2d_multiply_leakyrelu_gelu_cuda(torch::Tensor x, torch::Tensor multiplier);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_multiply_leakyrelu_gelu_cuda", &conv2d_multiply_leakyrelu_gelu_cuda, "Fused multiply+leakyrelu+gelu CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_multiply_leakyrelu_gelu_inplace_kernel(
    float* __restrict__ x_inout,
    const float* __restrict__ multiplier,
    int total,
    int HW,
    int C,
    int mult_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = (idx / HW) % C;
    int m_idx = (mult_numel == 1) ? 0 : c;
    float v = x_inout[idx] * __ldg(&multiplier[m_idx]);

    // Branchless LeakyReLU with slope 0.01
    v = fmaxf(v, 0.01f * v);

    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));

    x_inout[idx] = v;
}

torch::Tensor conv2d_multiply_leakyrelu_gelu_cuda(torch::Tensor x, torch::Tensor multiplier) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "multiplier must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto mult_flat = multiplier.contiguous().flatten();
    int mult_numel = (int)mult_flat.numel();

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;
    int total = N * C * HW;

    TORCH_CHECK(mult_numel == 1 || mult_numel == C,
        "multiplier numel must be 1 or C, got ", mult_numel);

    int block = 256;
    int grid = (total + block - 1) / block;

    fused_multiply_leakyrelu_gelu_inplace_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        mult_flat.data_ptr<float>(),
        total, HW, C, mult_numel
    );

    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
        """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _fused = (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and self.multiplier.is_cuda)
        if _fused:
            x = _stark_get_extension().conv2d_multiply_leakyrelu_gelu_cuda(x, self.multiplier)
        else:
            x = x * self.multiplier
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not _fused:
                    x = self.leaky_relu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not _fused:
                    x = torch.nn.functional.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
