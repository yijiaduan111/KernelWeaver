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

# <<<IMPROVE:user_helpers>>>
# <<<END_IMPROVE>>>

CUDA_CPP_SRC = r"""
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>

torch::Tensor swish_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish_cuda", &swish_cuda, "Swish CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void swish_kernel_vec2(const float2* __restrict__ x, float2* __restrict__ out, int64_t vec_elems) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < vec_elems; i += stride) {
        float2 v = __ldg(&x[i]);
        float2 r;
        r.x = v.x / (1.0f + expf(-v.x));
        r.y = v.y / (1.0f + expf(-v.y));
        out[i] = r;
    }
}

__global__ void swish_kernel_scalar(const float* __restrict__ x, float* __restrict__ out, int64_t offset, int64_t numel) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float v = x[offset + i];
        out[offset + i] = v / (1.0f + expf(-v));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "swish_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "swish_cuda: input must be float32");
    auto x_contig = x.contiguous();
    auto out = torch::empty_like(x_contig);
    int64_t numel = x_contig.numel();
    const int threads = 256;

    float* xptr = x_contig.data_ptr<float>();
    float* optr = out.data_ptr<float>();
    bool aligned = (((uintptr_t)xptr % 8) == 0) && (((uintptr_t)optr % 8) == 0);

    if (aligned && numel >= 2) {
        int64_t vec_elems = numel / 2;
        int64_t tail = numel % 2;
        int blocks_vec = (int)std::min((vec_elems + threads - 1) / threads, (int64_t)65535);
        swish_kernel_vec2<<<blocks_vec, threads>>>(
            reinterpret_cast<const float2*>(xptr),
            reinterpret_cast<float2*>(optr),
            vec_elems
        );
        if (tail > 0) {
            int64_t offset = vec_elems * 2;
            swish_kernel_scalar<<<1, (int)tail>>>(xptr, optr, offset, tail);
        }
    } else {
        int blocks = (int)std::min((numel + threads - 1) / threads, (int64_t)65535);
        swish_kernel_scalar<<<blocks, threads>>>(xptr, optr, 0, numel);
    }
    return out.view(x.sizes());
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a Swish activation.
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
                Applies Swish activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of any shape.

                Returns:
                    torch.Tensor: Output tensor with Swish applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32:
            return _stark_get_extension().swish_cuda(x)
        return x * torch.sigmoid(x)
        # <<<END_IMPROVE>>>
