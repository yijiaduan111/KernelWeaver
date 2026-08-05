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
    return f'stark_cuda_l2_p39_{digest}'

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

torch::Tensor scale_mul_cuda(torch::Tensor x, torch::Tensor scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_mul_cuda", &scale_mul_cuda, "Vectorized per-channel scale multiply (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Row-wise in-place scale kernel: one block per row, grid-stride over channels
__global__ void scale_mul_rowwise_kernel(
    float* __restrict__ x,
    const float* __restrict__ scale,
    int N, int C
) {
    int row = blockIdx.x;
    if (row >= N) return;
    float* row_ptr = x + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        row_ptr[c] *= scale[c];
    }
}

// Row-wise float4 in-place kernel for aligned channels
__global__ void scale_mul_rowwise_kernel_f4(
    float* __restrict__ x,
    const float* __restrict__ scale,
    int N, int C
) {
    int row = blockIdx.x;
    if (row >= N) return;
    float* row_ptr = x + row * C;
    int C4 = C / 4;
    for (int c4 = threadIdx.x; c4 < C4; c4 += blockDim.x) {
        float4 val = reinterpret_cast<float4*>(row_ptr)[c4];
        int base = c4 * 4;
        val.x *= scale[base];
        val.y *= scale[base + 1];
        val.z *= scale[base + 2];
        val.w *= scale[base + 3];
        reinterpret_cast<float4*>(row_ptr)[c4] = val;
    }
    // Scalar tail for C % 4 != 0
    for (int c = C4 * 4 + threadIdx.x; c < C; c += blockDim.x) {
        row_ptr[c] *= scale[c];
    }
}

torch::Tensor scale_mul_cuda(torch::Tensor x, torch::Tensor scale) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(scale.dim() == 1, "scale must be 1D");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    int N = x.size(0);
    int C = x.size(1);
    TORCH_CHECK(scale.numel() == C, "scale.numel() must equal x.size(1)");

    constexpr int BLOCK = 256;
    int grid = N;

    if (C % 4 == 0 && C >= 4) {
        scale_mul_rowwise_kernel_f4<<<grid, BLOCK>>>(
            x.data_ptr<float>(),
            scale.data_ptr<float>(),
            N, C
        );
    } else {
        scale_mul_rowwise_kernel<<<grid, BLOCK>>>(
            x.data_ptr<float>(),
            scale.data_ptr<float>(),
            N, C
        );
    }

    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
        """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _scale_cuda = (
            x.is_cuda and
            x.dtype == torch.float32 and
            x.dim() == 2 and
            x.is_contiguous() and
            self.scale.dim() == 1 and
            self.scale.is_cuda and
            self.scale.dtype == torch.float32 and
            self.scale.is_contiguous() and
            self.scale.numel() == x.size(1)
        )
        if _scale_cuda:
            x = _stark_get_extension().scale_mul_cuda(x, self.scale)
        else:
            x = x * self.scale
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.bn(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
