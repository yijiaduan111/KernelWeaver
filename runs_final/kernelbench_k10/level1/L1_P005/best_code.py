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
    return f'stark_cuda_l1_p5_{digest}'

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

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor a, double s);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_scalar_mul_cuda", &matrix_scalar_mul_cuda, "matrix scalar multiply (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void scalar_mul_kernel_float4(const float4* __restrict__ in, float4* __restrict__ out, float s, int64_t n4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n4; i += stride) {
        float4 v = in[i];
        v.x *= s;
        v.y *= s;
        v.z *= s;
        v.w *= s;
        out[i] = v;
    }
}

__global__ void scalar_mul_kernel(const float* __restrict__ in, float* __restrict__ out, float s, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = in[i] * s;
    }
}

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor a, double s) {
    TORCH_CHECK(a.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(a.is_contiguous(), "Input must be contiguous");

    auto out = torch::empty_like(a);
    int64_t n = a.numel();
    float fs = static_cast<float>(s);

    const int threads = 256;
    int64_t n4 = n / 4;
    int64_t rem = n % 4;

    if (n4 > 0) {
        int blocks = static_cast<int>(std::min(n4, (int64_t)65535));
        scalar_mul_kernel_float4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(a.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            fs, n4);
    }
    if (rem > 0) {
        // handle tail elements
        scalar_mul_kernel<<<1, rem>>>(
            a.data_ptr<float>() + n4 * 4,
            out.data_ptr<float>() + n4 * 4,
            fs, rem);
    }
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a matrix-scalar multiplication (C = A * s)
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs matrix-scalar multiplication.

                Args:
                    A: Input matrix of shape (M, N)
                    s: Scalar value

                Returns:
                    C: Resulting matrix of shape (M, N)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if A.is_cuda and A.dtype == torch.float32 and A.is_contiguous():
            return _stark_get_extension().matrix_scalar_mul_cuda(A, float(s))
        return A * s
        # <<<END_IMPROVE>>>
