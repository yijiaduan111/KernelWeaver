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
    return f'stark_cuda_l2_p86_{digest}'

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

torch::Tensor divide_gelu_cuda(torch::Tensor x, double divisor);

torch::Tensor divide_gelu(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "divide_gelu: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "divide_gelu: x must be float32");
    return divide_gelu_cuda(x, divisor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("divide_gelu", &divide_gelu, "Fused divide + GELU (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void __launch_bounds__(256)
divide_gelu_kernel(float* __restrict__ data,
                   int numel,
                   float inv_divisor) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < numel) {
        float4 v = reinterpret_cast<float4*>(data)[idx / 4];
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float s = vals[i] * inv_divisor;
            float cube = s * s * s;
            vals[i] = 0.5f * s * (1.0f + tanhf(0.7978845608f * (s + 0.044715f * cube)));
        }
        float4 out;
        out.x = vals[0]; out.y = vals[1]; out.z = vals[2]; out.w = vals[3];
        reinterpret_cast<float4*>(data)[idx / 4] = out;
    } else {
        for (int i = idx; i < numel && i < idx + 4; i++) {
            float s = data[i] * inv_divisor;
            float cube = s * s * s;
            data[i] = 0.5f * s * (1.0f + tanhf(0.7978845608f * (s + 0.044715f * cube)));
        }
    }
}

torch::Tensor divide_gelu_cuda(torch::Tensor x, double divisor) {
    auto x_contig = x.contiguous();
    int numel = x_contig.numel();
    float inv_divisor = 1.0f / static_cast<float>(divisor);
    int threads = 256;
    int blocks = (numel + threads * 4 - 1) / (threads * 4);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    divide_gelu_kernel<<<blocks, threads, 0, stream>>>(
        x_contig.data_ptr<float>(),
        numel,
        inv_divisor
    );
    return x_contig;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
        """
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, input_size).
                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, output_size).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.linear(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if x.is_cuda and x.dtype == torch.float32:
            x = _stark_get_extension().divide_gelu(x, float(self.divisor))
        else:
            x = x / self.divisor
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not (x.is_cuda and x.dtype == torch.float32):
                    x = torch.nn.functional.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
