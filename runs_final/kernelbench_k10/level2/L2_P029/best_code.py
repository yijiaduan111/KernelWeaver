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
    return f'stark_cuda_l2_p29_{digest}'

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

torch::Tensor double_mish_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("double_mish_cuda", &double_mish_cuda, "Fused double Mish (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float mish_activation(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void double_mish_inplace_vec(float* data, int numel) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < numel) {
        float4 val = *reinterpret_cast<float4*>(data + idx);
        val.x = mish_activation(mish_activation(val.x));
        val.y = mish_activation(mish_activation(val.y));
        val.z = mish_activation(mish_activation(val.z));
        val.w = mish_activation(mish_activation(val.w));
        *reinterpret_cast<float4*>(data + idx) = val;
    }
}

__global__ void double_mish_inplace_scalar(float* data, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = data[idx];
        data[idx] = mish_activation(mish_activation(val));
    }
}

torch::Tensor double_mish_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int numel = x.numel();
    const int threads = 256;
    float* data_ptr = x.data_ptr<float>();

    bool is_aligned = (reinterpret_cast<uintptr_t>(data_ptr) % 16 == 0);

    if (is_aligned && numel >= 4) {
        int vec_numel = (numel / 4) * 4;
        int vec_blocks = (vec_numel / 4 + threads - 1) / threads;
        double_mish_inplace_vec<<<vec_blocks, threads>>>(data_ptr, numel);

        int tail = numel - vec_numel;
        if (tail > 0) {
            int tail_blocks = (tail + threads - 1) / threads;
            double_mish_inplace_scalar<<<tail_blocks, threads>>>(data_ptr + vec_numel, tail);
        }
    } else {
        int blocks = (numel + threads - 1) / threads;
        double_mish_inplace_scalar<<<blocks, threads>>>(data_ptr, numel);
    }

    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
        """
    def __init__(self, in_features, out_features):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.linear = nn.Linear(in_features, out_features)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.linear(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().double_mish_cuda(x.contiguous())
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # Second Mish already applied in fused kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
