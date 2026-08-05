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
    return f'stark_cuda_l2_p63_{digest}'

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

torch::Tensor relu_divide_cuda(torch::Tensor x, double divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_divide_cuda", &relu_divide_cuda, "Fused ReLU+divide CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void relu_divide_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float inv_divisor,
    int64_t n_vec4)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        float4 v = ((const float4*)input)[idx];
        v.x = fmaxf(v.x, 0.0f) * inv_divisor;
        v.y = fmaxf(v.y, 0.0f) * inv_divisor;
        v.z = fmaxf(v.z, 0.0f) * inv_divisor;
        v.w = fmaxf(v.w, 0.0f) * inv_divisor;
        ((float4*)output)[idx] = v;
    }
}

__global__ void relu_divide_scalar_tail_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float inv_divisor,
    int64_t offset,
    int64_t total)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < total) {
        output[idx] = fmaxf(input[idx], 0.0f) * inv_divisor;
    }
}

torch::Tensor relu_divide_cuda(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "relu_divide_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "relu_divide_cuda: input must be float32");
    auto x_c = x.contiguous();
    auto out = torch::empty_like(x_c);
    int64_t numel = x_c.numel();
    float inv_divisor = 1.0f / (float)divisor;
    const int BLOCK = 256;
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t n_vec4 = numel / 4;
    int64_t remainder = numel % 4;
    if (n_vec4 > 0) {
        int64_t grid = (n_vec4 + BLOCK - 1) / BLOCK;
        relu_divide_vec4_kernel<<<grid, BLOCK, 0, stream>>>(
            x_c.data_ptr<float>(), out.data_ptr<float>(), inv_divisor, n_vec4);
    }
    if (remainder > 0) {
        relu_divide_scalar_tail_kernel<<<1, BLOCK, 0, stream>>>(
            x_c.data_ptr<float>(), out.data_ptr<float>(), inv_divisor,
            n_vec4 * 4, numel);
    }
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a matrix multiplication, applies ReLU, and divides by a constant.
        """
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.linear(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().relu_divide_cuda(x, float(self.divisor))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # fused into relu_divide_cuda above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
