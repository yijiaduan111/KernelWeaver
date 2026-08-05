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
    return f'stark_cuda_l2_p1_{digest}'

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

torch::Tensor relu_bias_add_inplace(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_bias_add_inplace", &relu_bias_add_inplace, "Fused ReLU + bias add (in-place, CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Slice-specialized scalar kernel: blockIdx.y = nc slice, threadIdx.x+blockIdx.x*blockDim.x = spatial pos
__global__ void relu_bias_add_slice_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int NC, int C, int HW
) {
    int nc = blockIdx.y;
    int spatial = blockIdx.x * blockDim.x + threadIdx.x;
    if (nc >= NC || spatial >= HW) return;
    int c = nc % C;
    float b = bias[c];
    int base = nc * HW + spatial;
    float v = x[base];
    x[base] = (v > 0.0f ? v : 0.0f) + b;
}

// Slice-specialized vec4 kernel: each thread handles 4 contiguous spatial elements within one (n,c) slice
__global__ void relu_bias_add_slice_vec4_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int NC, int C, int HW4
) {
    int nc = blockIdx.y;
    int spatial4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (nc >= NC || spatial4 >= HW4) return;
    int c = nc % C;
    float b = __ldg(&bias[c]);
    float4* xv = reinterpret_cast<float4*>(x) + nc * HW4 + spatial4;
    float4 v = *xv;
    v.x = (v.x > 0.0f ? v.x : 0.0f) + b;
    v.y = (v.y > 0.0f ? v.y : 0.0f) + b;
    v.z = (v.z > 0.0f ? v.z : 0.0f) + b;
    v.w = (v.w > 0.0f ? v.w : 0.0f) + b;
    *xv = v;
}

torch::Tensor relu_bias_add_inplace(torch::Tensor x, torch::Tensor bias) {
    int N  = x.size(0);
    int C  = x.size(1);
    int H  = x.size(2);
    int W  = x.size(3);
    int HW = H * W;
    int NC = N * C;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool aligned16 = (reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16 == 0);

    if (aligned16 && HW % 4 == 0) {
        int HW4 = HW / 4;
        int threads = 256;
        int bx = (HW4 + threads - 1) / threads;
        dim3 grid(bx, NC);
        relu_bias_add_slice_vec4_kernel<<<grid, threads, 0, stream>>>(
            x.data_ptr<float>(),
            bias.data_ptr<float>(),
            NC, C, HW4
        );
    } else {
        int threads = 256;
        int bx = (HW + threads - 1) / threads;
        dim3 grid(bx, NC);
        relu_bias_add_slice_kernel<<<grid, threads, 0, stream>>>(
            x.data_ptr<float>(),
            bias.data_ptr<float>(),
            NC, C, HW
        );
    }
    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a convolution, applies ReLU, and adds a bias term.
        """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            bias_flat = self.bias.contiguous().view(-1)
            x = _stark_get_extension().relu_bias_add_inplace(x, bias_flat)
        else:
            x = torch.relu(x)
            x = x + self.bias
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # bias add handled in forward_stmt_2 for the fast path
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
