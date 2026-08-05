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
    return f'stark_cuda_l2_p25_{digest}'

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

torch::Tensor min_tanh_tanh_cuda(torch::Tensor x);

torch::Tensor min_tanh_tanh(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "min_tanh_tanh: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "min_tanh_tanh: input must be float32");
    TORCH_CHECK(x.dim() == 4, "min_tanh_tanh: input must be 4D (N,C,H,W)");
    return min_tanh_tanh_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_tanh_tanh", &min_tanh_tanh, "Fused channel-min + tanh + tanh (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Each block handles a contiguous tile of HW positions.
// Threads within a block each handle one spatial position independently.
// This avoids division/modulo in the inner loop and keeps register use low.
__global__ void min_tanh_tanh_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int HW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * HW;
    if (idx >= total) return;

    int n  = idx / HW;
    int hw = idx - n * HW;
    // base points to channel 0 for this (n, hw) position in NCHW layout
    const float* ptr = input + n * (C * HW) + hw;

    float min_val = ptr[0];
    #pragma unroll 4
    for (int c = 1; c < C; ++c) {
        float v = ptr[c * HW];
        min_val = v < min_val ? v : min_val;
    }

    output[idx] = tanhf(tanhf(min_val));
}

torch::Tensor min_tanh_tanh_cuda(torch::Tensor x) {
    x = x.contiguous();
    int N  = x.size(0);
    int C  = x.size(1);
    int H  = x.size(2);
    int W  = x.size(3);
    int HW = H * W;

    auto output = torch::empty({N, 1, H, W}, x.options());

    int total   = N * HW;
    int threads = 512;
    int blocks  = (total + threads - 1) / threads;
    // stay inside grid dim limits
    if (blocks > 65535) blocks = 65535;

    min_tanh_tanh_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, HW
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
        """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            x = _stark_get_extension().min_tanh_tanh(x)
        else:
            x = torch.tanh(torch.tanh(torch.min(x, dim=1, keepdim=True)[0]))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # tanh already applied by fused kernel (or fallback in forward_stmt_2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # second tanh already applied by fused kernel (or fallback in forward_stmt_2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
