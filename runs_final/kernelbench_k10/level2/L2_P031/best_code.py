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
    return f'stark_cuda_l2_p31_{digest}'

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

torch::Tensor conv2d_min_add_multiply_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double constant_value,
    double scaling_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_min_add_multiply_cuda", &conv2d_min_add_multiply_cuda,
          "Fused min+bias_add+scale CUDA pointwise kernel");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_min_bias_scale_3d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ bias,
    float constant_value,
    float scaling_factor,
    int hw,
    int C
) {
    int spatial_base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int c = blockIdx.y;
    int n = blockIdx.z;

    if (spatial_base >= hw) return;

    float b = __ldg(&bias[c]);
    int base_idx = ((n * C + c) * hw) + spatial_base;

    if (spatial_base + 1 < hw) {
        float2 v;
        v.x = __ldg(&x[base_idx]);
        v.y = __ldg(&x[base_idx + 1]);
        v.x = (fminf(v.x, constant_value) + b) * scaling_factor;
        v.y = (fminf(v.y, constant_value) + b) * scaling_factor;
        out[base_idx] = v.x;
        out[base_idx + 1] = v.y;
    } else {
        float v = __ldg(&x[base_idx]);
        out[base_idx] = (fminf(v, constant_value) + b) * scaling_factor;
    }
}

torch::Tensor conv2d_min_add_multiply_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double constant_value,
    double scaling_factor
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");

    x = x.contiguous();
    bias = bias.contiguous();

    auto out = torch::empty_like(x);

    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));
    int hw = H * W;

    float cv = static_cast<float>(constant_value);
    float sf = static_cast<float>(scaling_factor);

    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();

    int threads = 256;
    dim3 grid(((hw + 1) / 2 + threads - 1) / threads, C, N);
    fused_min_bias_scale_3d_kernel<<<grid, threads>>>(
        x_ptr, out_ptr, bias_ptr, cv, sf, hw, C
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
        """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().conv2d_min_add_multiply_cuda(x, self.bias, self.constant_value, self.scaling_factor)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x  # bias add fused into CUDA extension above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = x  # scale multiply fused into CUDA extension above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
