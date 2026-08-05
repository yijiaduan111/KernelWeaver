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
    return f'stark_cuda_l2_p47_{digest}'

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

torch::Tensor mish_tanh_inplace_cuda(torch::Tensor x);

torch::Tensor mish_tanh_inplace(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mish_tanh_inplace requires CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "mish_tanh_inplace requires float32");
    return mish_tanh_inplace_cuda(x.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mish_tanh_inplace", &mish_tanh_inplace, "Fused Mish+Tanh inplace (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float mish_tanh_eval(float v) {
    if (v > 20.0f) {
        return tanhf(v);
    } else if (v > 8.0f) {
        // softplus(v) ~ v for large v, so mish(v) ~ v*tanh(v)
        return tanhf(v * tanhf(v));
    } else if (v < -10.0f) {
        return 0.0f;
    } else {
        float sp = log1pf(expf(v));
        return tanhf(v * tanhf(sp));
    }
}

__global__ void mish_tanh_kernel(float* __restrict__ data, int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = mish_tanh_eval(data[idx]);
    }
}

torch::Tensor mish_tanh_inplace_cuda(torch::Tensor x) {
    int64_t numel = x.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);

    mish_tanh_kernel<<<blocks, threads>>>(x.data_ptr<float>(), numel);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mish_tanh_kernel launch failed: ", cudaGetErrorString(err));

    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if x.is_cuda and x.dtype == torch.float32:
            x = _stark_get_extension().mish_tanh_inplace(x)
        else:
            x = torch.nn.functional.mish(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not (x.is_cuda and x.dtype == torch.float32):
            x = torch.tanh(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
