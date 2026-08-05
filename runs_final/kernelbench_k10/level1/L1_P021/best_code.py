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
    return f'stark_cuda_l1_p21_{digest}'

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

torch::Tensor sigmoid_cuda(torch::Tensor x);

torch::Tensor sigmoid_dispatch(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sigmoid_dispatch: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "sigmoid_dispatch: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "sigmoid_dispatch: input must be contiguous");
    return sigmoid_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_cuda", &sigmoid_dispatch, "Sigmoid CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel_float4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t n4,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);

    for (int64_t i = idx; i < n4; i += stride) {
        float4 val = in4[i];
        val.x = fmaf(0.5f, __tanhf(0.5f * val.x), 0.5f);
        val.y = fmaf(0.5f, __tanhf(0.5f * val.y), 0.5f);
        val.z = fmaf(0.5f, __tanhf(0.5f * val.z), 0.5f);
        val.w = fmaf(0.5f, __tanhf(0.5f * val.w), 0.5f);
        out4[i] = val;
    }

    // Handle tail elements
    int64_t tail_start = n4 * 4;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        output[i] = fmaf(0.5f, __tanhf(0.5f * input[i]), 0.5f);
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    int64_t n4 = n / 4;

    const int threads = 256;
    int64_t blocks = std::min((n4 + threads - 1) / threads, (int64_t)65535);
    if (blocks == 0) blocks = 1;

    sigmoid_kernel_float4<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        n4,
        n
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a Sigmoid activation.
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies Sigmoid activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of any shape.

                Returns:
                    torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return x.sigmoid()
        # <<<END_IMPROVE>>>
