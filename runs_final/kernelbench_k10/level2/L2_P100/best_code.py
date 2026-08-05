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
    return f'stark_cuda_l2_p100_{digest}'

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

void clamp_div_inplace(torch::Tensor x, double min_value, double inv_div);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("clamp_div_inplace", &clamp_div_inplace, "Fused in-place clamp(min) then multiply by reciprocal (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void clamp_div_inplace_kernel_f4(
    float4* __restrict__ data,
    int64_t n4,
    float min_val,
    float inv_div
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 v = data[idx];
        v.x = fmaxf(v.x, min_val) * inv_div;
        v.y = fmaxf(v.y, min_val) * inv_div;
        v.z = fmaxf(v.z, min_val) * inv_div;
        v.w = fmaxf(v.w, min_val) * inv_div;
        data[idx] = v;
    }
}

__global__ void clamp_div_inplace_kernel_scalar(
    float* __restrict__ data,
    int64_t n,
    float min_val,
    float inv_div
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], min_val) * inv_div;
    }
}

void clamp_div_inplace(torch::Tensor x, double min_value, double inv_div) {
    TORCH_CHECK(x.device().is_cuda(), "clamp_div_inplace: input must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "clamp_div_inplace: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "clamp_div_inplace: input must be contiguous");
    int64_t n = x.numel();
    float min_val = static_cast<float>(min_value);
    float inv_d = static_cast<float>(inv_div);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 128;
    int64_t n4 = n / 4;
    int64_t tail = n % 4;
    if (n4 > 0) {
        int64_t blocks = (n4 + threads - 1) / threads;
        clamp_div_inplace_kernel_f4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4*>(x.data_ptr<float>()),
            n4, min_val, inv_d
        );
    }
    if (tail > 0) {
        int64_t offset = n4 * 4;
        int64_t blocks = (tail + threads - 1) / threads;
        clamp_div_inplace_kernel_scalar<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>() + offset,
            tail, min_val, inv_d
        );
    }
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
        and then divides the result by a constant.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if not x.is_contiguous():
            x = x.contiguous()
        _stark_get_extension().clamp_div_inplace(x, float(self.min_value), float(1.0 / self.divisor))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
