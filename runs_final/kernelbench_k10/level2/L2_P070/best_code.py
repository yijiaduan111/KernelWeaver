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
    return f'stark_cuda_l2_p70_{digest}'

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

torch::Tensor sigmoid_scale_residual_cuda(torch::Tensor x, double scaling_factor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_scale_residualadd", &sigmoid_scale_residual_cuda, "Fused sigmoid + scale + residual add (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void __launch_bounds__(256, 8)
sigmoid_scale_residual_kernel_vec4(float4* __restrict__ data,
                                    int numel4,
                                    float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < numel4; i += stride) {
        float4 v = data[i];
        float s0 = 1.0f / (1.0f + __expf(-v.x));
        float s1 = 1.0f / (1.0f + __expf(-v.y));
        float s2 = 1.0f / (1.0f + __expf(-v.z));
        float s3 = 1.0f / (1.0f + __expf(-v.w));
        float4 r;
        r.x = s0 * scaling_factor + v.x;
        r.y = s1 * scaling_factor + v.y;
        r.z = s2 * scaling_factor + v.z;
        r.w = s3 * scaling_factor + v.w;
        data[i] = r;
    }
}

__global__ void __launch_bounds__(256, 8)
sigmoid_scale_residual_kernel_scalar(float* __restrict__ data,
                                      int numel,
                                      float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < numel; i += stride) {
        float v = data[i];
        float s = 1.0f / (1.0f + __expf(-v));
        data[i] = s * scaling_factor + v;
    }
}

torch::Tensor sigmoid_scale_residual_cuda(torch::Tensor x, double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    int numel = x.numel();
    float sf = static_cast<float>(scaling_factor);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(x.get_device());

    const int BLOCK = 256;
    if (numel % 4 == 0) {
        int numel4 = numel / 4;
        int grid = (numel4 + BLOCK - 1) / BLOCK;
        grid = min(grid, 65535);
        sigmoid_scale_residual_kernel_vec4<<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<float4*>(x.data_ptr<float>()),
            numel4, sf);
    } else {
        int grid = (numel + BLOCK - 1) / BLOCK;
        grid = min(grid, 65535);
        sigmoid_scale_residual_kernel_scalar<<<grid, BLOCK, 0, stream>>>(
            x.data_ptr<float>(), numel, sf);
    }
    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
        """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the model.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, input_size).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, hidden_size).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().sigmoid_scale_residualadd(x, float(self.scaling_factor))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # sigmoid handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # scaling handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # residual add handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
