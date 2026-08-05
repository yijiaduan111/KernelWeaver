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
    return f'stark_cuda_l1_p30_{digest}'

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

torch::Tensor softsign_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softsign_cuda", &softsign_cuda, "Fused Softsign activation (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__
void softsign_kernel_f4_aligned(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);
    for (int i = idx; i < n4; i += stride) {
        float4 v = __ldg(&in4[i]);
        v.x = __fdividef(v.x, 1.0f + fabsf(v.x));
        v.y = __fdividef(v.y, 1.0f + fabsf(v.y));
        v.z = __fdividef(v.z, 1.0f + fabsf(v.z));
        v.w = __fdividef(v.w, 1.0f + fabsf(v.w));
        out4[i] = v;
    }
}

__global__
void softsign_kernel_scalar(const float* __restrict__ input,
                             float* __restrict__ output,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float v = __ldg(&input[i]);
        output[i] = __fdividef(v, 1.0f + fabsf(v));
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "softsign_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "softsign_cuda: input must be float32");
    auto x_cont = x.contiguous();
    auto output = torch::empty_like(x_cont);
    int n = x_cont.numel();
    const int block = 512;
    auto stream = at::cuda::getDefaultCUDAStream();
    bool aligned = ((reinterpret_cast<uintptr_t>(x_cont.data_ptr<float>()) & 15) == 0) &&
                   ((reinterpret_cast<uintptr_t>(output.data_ptr<float>()) & 15) == 0) &&
                   ((n & 3) == 0);
    if (aligned) {
        int n4 = n / 4;
        int grid = (n4 + block - 1) / block;
        if (grid == 0) grid = 1;
        softsign_kernel_f4_aligned<<<grid, block, 0, stream>>>(
            x_cont.data_ptr<float>(),
            output.data_ptr<float>(),
            n4
        );
    } else {
        int grid = (n + block - 1) / block;
        if (grid == 0) grid = 1;
        softsign_kernel_scalar<<<grid, block, 0, stream>>>(
            x_cont.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );
    }
    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a Softsign activation.
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
                Applies Softsign activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of any shape.

                Returns:
                    torch.Tensor: Output tensor with Softsign applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return _stark_get_extension().softsign_cuda(x)
        return x / (1 + torch.abs(x))
        # <<<END_IMPROVE>>>
