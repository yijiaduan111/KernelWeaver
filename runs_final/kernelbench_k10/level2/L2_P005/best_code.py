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
    return f'stark_cuda_l2_p5_{digest}'

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

torch::Tensor fused_subtract_tanh(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_subtract_tanh", &fused_subtract_tanh, "Fused subtract+bias tanh (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

// 3D grid: z=n, y=c, x=spatial_block
// bias[c] is read directly from blockIdx.y - no per-thread modulo needed
// In-place: single buffer read-modify-write, safe because each element is independent
__global__ void fused_subtract_tanh_nc_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int64_t spatial,
    int64_t C) {
    int64_t n = static_cast<int64_t>(blockIdx.z);
    int64_t c = static_cast<int64_t>(blockIdx.y);
    float b = bias[c];  // one register load per plane, no division/modulo
    int64_t base = (n * C + c) * spatial;
    // grid-stride loop covers full spatial dimension
    for (int64_t s = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         s < spatial;
         s += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        x[base + s] = tanhf(x[base + s] - b);
    }
}

}  // namespace

torch::Tensor fused_subtract_tanh(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(bias.numel() == x.size(1), "bias must have one value per channel");

    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous().view({x.size(1)});

    const int64_t N = x_contig.size(0);
    const int64_t C = x_contig.size(1);
    const int64_t spatial = x_contig.size(2) * x_contig.size(3);

    const c10::cuda::CUDAGuard device_guard(x_contig.device());
    constexpr int threads = 256;
    const int blocks_x = static_cast<int>((spatial + threads - 1) / threads);
    // 3D grid: z=N batch, y=C channels, x=spatial tiles
    dim3 grid(blocks_x, static_cast<int>(C), static_cast<int>(N));
    fused_subtract_tanh_nc_kernel<<<grid, threads>>>(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        spatial,
        C);

    return x_contig;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _stark_can_fuse = (
            x.is_cuda and
            x.dtype == torch.float32 and
            self.bias.is_cuda and
            self.bias.dtype == torch.float32 and
            x.dim() == 4 and
            x.is_contiguous() and
            self.bias.is_contiguous() and
            self.bias.numel() == x.shape[1]
        )
        if _stark_can_fuse:
            x = _stark_get_extension().fused_subtract_tanh(x, self.bias)
        else:
            x = x - self.bias
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not _stark_can_fuse:
            x = torch.tanh(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
