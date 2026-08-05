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
    return f'stark_cuda_l2_p32_{digest}'

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

torch::Tensor fused_scale_min_cuda(torch::Tensor input, float scale_factor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scale_min", &fused_scale_min_cuda, "Fused scale and channel min (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_scale_min_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const float scale_factor
) {
    const int spatial_size = H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N * spatial_size) return;

    const int n = idx / spatial_size;
    const int hw = idx % spatial_size;
    const int h = hw / W;
    const int w = hw % W;

    float min_val = INFINITY;

    for (int c = 0; c < C; ++c) {
        const int input_idx = n * C * H * W + c * H * W + h * W + w;
        const float scaled_val = input[input_idx] * scale_factor;
        min_val = fminf(min_val, scaled_val);
    }

    output[idx] = min_val;
}

torch::Tensor fused_scale_min_cuda(torch::Tensor input, float scale_factor) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (NCHW)");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty({N, 1, H, W}, input.options());

    const int spatial_size = H * W;
    const int total_pixels = N * spatial_size;
    const int threads = 256;
    const int blocks = (total_pixels + threads - 1) / threads;

    fused_scale_min_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        scale_factor
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, scales the output, and then applies a minimum operation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().fused_scale_min(x, float(self.scale_factor))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # Min reduction now fused into the custom kernel above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
