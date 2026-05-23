import torch
import torch.nn as nn
import hashlib
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
    return f'stark_cuda_l1_p20_{digest}'

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

CUDA_CPP_SRC = r"""
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>

torch::Tensor leaky_relu_cuda(torch::Tensor x, double negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_cuda", &leaky_relu_cuda, "LeakyReLU CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int n,
                                   float negative_slope) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float v = input[i];
        output[i] = v >= 0.0f ? v : v * negative_slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "leaky_relu_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "leaky_relu_cuda: only float32 is supported");

    auto x_contig = x.contiguous();
    auto out = torch::empty_like(x_contig);

    int n = (int)x_contig.numel();
    if (n == 0) {
        return out.view(x.sizes());
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        (float)negative_slope
    );

    return out.view(x.sizes());
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a LeakyReLU activation.
        """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the LeakyReLU module.

                Args:
                    negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
                """
        self.negative_slope = negative_slope
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies LeakyReLU activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of any shape.

                Returns:
                    torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return _stark_get_extension().leaky_relu_cuda(x, float(self.negative_slope))
        return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        # <<<END_IMPROVE>>>