import hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# <<<IMPROVE:helpers>>>
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
    return f'stark_cuda_l1_p25_{digest}'

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
# <<<END_IMPROVE>>>

CUDA_CPP_SRC = r'''
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>

torch::Tensor swish_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish_cuda", &swish_cuda, "Fused Swish CUDA");
}
# <<<END_IMPROVE>>>
'''

CUDA_CU_SRC = r'''
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

__global__ void swish_kernel(const float* x, float* y, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float v = x[i];
        y[i] = v / (1.0f + __expf(-v));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return y;
    const int threads = 256;
    const int blocks = std::min<int64_t>((n + threads - 1) / threads, 4096);
    swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
# <<<END_IMPROVE>>>
'''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
        Applies Swish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return _stark_get_extension().swish_cuda(x.contiguous()) if x.is_cuda and x.dtype == torch.float32 else x * torch.sigmoid(x)
        # <<<END_IMPROVE>>>