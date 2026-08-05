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
    return f'stark_cuda_l2_p4_{digest}'

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

void double_mish_inplace_cuda(torch::Tensor x);

void double_mish_inplace(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "double_mish_inplace: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "double_mish_inplace: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "double_mish_inplace: input must be contiguous");
    double_mish_inplace_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("double_mish_inplace", &double_mish_inplace, "Fused in-place double Mish (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float mish_scalar(float x) {
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = __expf(x);
    } else {
        sp = __logf(1.0f + __expf(x));
    }
    return x * tanhf(sp);
}

__global__ void __launch_bounds__(256, 4)
double_mish_inplace_kernel(float* __restrict__ data, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = data[i];
        data[i] = mish_scalar(mish_scalar(x));
    }
}

__global__ void __launch_bounds__(256, 4)
double_mish_inplace_vec4_kernel(float4* __restrict__ data, int64_t n4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n4; i += stride) {
        float4 v = data[i];
        v.x = mish_scalar(mish_scalar(v.x));
        v.y = mish_scalar(mish_scalar(v.y));
        v.z = mish_scalar(mish_scalar(v.z));
        v.w = mish_scalar(mish_scalar(v.w));
        data[i] = v;
    }
}

void double_mish_inplace_cuda(torch::Tensor x) {
    int64_t n = x.numel();
    const int block = 256;
    if (n % 4 == 0) {
        int64_t n4 = n / 4;
        int grid = (int)((n4 + block - 1) / block);
        if (grid > 65535) grid = 65535;
        double_mish_inplace_vec4_kernel<<<grid, block>>>(
            reinterpret_cast<float4*>(x.data_ptr<float>()),
            n4);
    } else {
        int grid = (int)((n + block - 1) / block);
        if (grid > 65535) grid = 65535;
        double_mish_inplace_kernel<<<grid, block>>>(x.data_ptr<float>(), n);
    }
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a convolution, applies Mish, and another Mish.
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
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and not x.requires_grad:
            _stark_get_extension().double_mish_inplace(x)
        else:
            x = torch.nn.functional.mish(x)
            x = torch.nn.functional.mish(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
