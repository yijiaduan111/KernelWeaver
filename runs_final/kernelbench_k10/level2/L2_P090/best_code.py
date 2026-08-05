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
    return f'stark_cuda_l2_p90_{digest}'

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

torch::Tensor fused_post_ops(torch::Tensor x, torch::Tensor sum_tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_post_ops", &fused_post_ops, "Fused LeakyReLU + add + clamp + GELU (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void fused_post_ops_kernel(float* x, const float* sum, int64_t numel, int64_t channels, int64_t spatial) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    int64_t c = (idx / spatial) % channels;
    float v = x[idx];
    v = v >= 0.0f ? v : 0.2f * v;
    v += sum[c];
    v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    x[idx] = v;
}

}  // namespace

torch::Tensor fused_post_ops(torch::Tensor x, torch::Tensor sum_tensor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(sum_tensor.is_cuda(), "sum_tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(sum_tensor.scalar_type() == at::kFloat, "sum_tensor must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be NCDHW");
    TORCH_CHECK(sum_tensor.numel() == x.size(1), "sum_tensor must provide one value per channel");

    auto out = x.contiguous();
    auto sum_flat = sum_tensor.contiguous().view({x.size(1)});

    const int64_t numel = out.numel();
    const int64_t channels = out.size(1);
    const int64_t spatial = out.size(2) * out.size(3) * out.size(4);

    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    fused_post_ops_kernel<<<blocks, threads>>>(out.data_ptr<float>(), sum_flat.data_ptr<float>(), numel, channels, spatial);

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and self.sum_tensor.is_cuda and self.sum_tensor.dtype == torch.float32:
            x = _stark_get_extension().fused_post_ops(x, self.sum_tensor)
        else:
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not (x.is_cuda and x.dtype == torch.float32 and self.sum_tensor.is_cuda and self.sum_tensor.dtype == torch.float32):
                    x = x + self.sum_tensor
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not (x.is_cuda and x.dtype == torch.float32 and self.sum_tensor.is_cuda and self.sum_tensor.dtype == torch.float32):
                    x = torch.clamp(x, min=-1.0, max=1.0)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if not (x.is_cuda and x.dtype == torch.float32 and self.sum_tensor.is_cuda and self.sum_tensor.dtype == torch.float32):
                    x = torch.nn.functional.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
