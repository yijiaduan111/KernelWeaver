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
    return f'stark_cuda_l1_p28_{digest}'

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

torch::Tensor hardsigmoid_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hardsigmoid_cuda", &hardsigmoid_cuda, "HardSigmoid CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hsig(float v) {
    return fminf(1.0f, fmaxf(0.0f, fmaf(v, 0.16666667f, 0.5f)));
}

__global__ void hardsigmoid_vec4_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int64_t vec_n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < vec_n; i += stride) {
        float4 v = __ldg(input + i);
        v.x = hsig(v.x);
        v.y = hsig(v.y);
        v.z = hsig(v.z);
        v.w = hsig(v.w);
        output[i] = v;
    }
}

__global__ void hardsigmoid_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t offset,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        output[offset + i] = hsig(input[offset + i]);
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t vec_n = n / 4;
    int64_t tail = n % 4;

    if (vec_n > 0) {
        const int block = 256;
        int64_t grid = (vec_n + block - 1) / block;
        if (grid > 65535) grid = 65535;
        hardsigmoid_vec4_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4*>(in_ptr),
            reinterpret_cast<float4*>(out_ptr),
            vec_n
        );
    }

    if (tail > 0) {
        int64_t offset = vec_n * 4;
        hardsigmoid_scalar_kernel<<<1, tail, 0, stream>>>(
            in_ptr, out_ptr, offset, tail
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a HardSigmoid activation.
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
                Applies HardSigmoid activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of any shape.

                Returns:
                    torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return torch.nn.functional.hardsigmoid(x)
        # <<<END_IMPROVE>>>
