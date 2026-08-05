import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return f'stark_cuda_l3_p8_{digest}'

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

torch::Tensor add_relu(torch::Tensor out, torch::Tensor identity);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_relu", &add_relu, "Residual add + ReLU (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void add_relu_inplace_scalar_kernel(float* __restrict__ out,
                                               const float* __restrict__ identity,
                                               int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        float v = out[i] + identity[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void add_relu_inplace_vec4_kernel(float4* __restrict__ out,
                                              const float4* __restrict__ identity,
                                              int numel4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel4; i += stride) {
        float4 va = out[i];
        float4 vb = identity[i];
        float4 vy;
        vy.x = va.x + vb.x; vy.x = vy.x > 0.0f ? vy.x : 0.0f;
        vy.y = va.y + vb.y; vy.y = vy.y > 0.0f ? vy.y : 0.0f;
        vy.z = va.z + vb.z; vy.z = vy.z > 0.0f ? vy.z : 0.0f;
        vy.w = va.w + vb.w; vy.w = vy.w > 0.0f ? vy.w : 0.0f;
        out[i] = vy;
    }
}

torch::Tensor add_relu(torch::Tensor out, torch::Tensor identity) {
    TORCH_CHECK(out.is_cuda() && identity.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(identity.scalar_type() == torch::kFloat32, "identity must be float32");
    TORCH_CHECK(out.is_contiguous() && identity.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(out.sizes() == identity.sizes(), "Tensor shapes must match");

    int numel = static_cast<int>(out.numel());
    float* out_ptr = out.data_ptr<float>();
    const float* id_ptr = identity.data_ptr<float>();

    bool aligned = ((reinterpret_cast<std::uintptr_t>(out_ptr) % 16) == 0) &&
                   ((reinterpret_cast<std::uintptr_t>(id_ptr) % 16) == 0) &&
                   ((numel % 4) == 0);

    const int threads = 256;
    if (aligned) {
        int numel4 = numel / 4;
        int blocks = (numel4 + threads - 1) / threads;
        blocks = blocks < 65535 ? blocks : 65535;
        add_relu_inplace_vec4_kernel<<<blocks, threads>>>(
            reinterpret_cast<float4*>(out_ptr),
            reinterpret_cast<const float4*>(id_ptr),
            numel4);
    } else {
        int blocks = (numel + threads - 1) / threads;
        blocks = blocks < 65535 ? blocks : 65535;
        add_relu_inplace_scalar_kernel<<<blocks, threads>>>(out_ptr, id_ptr, numel);
    }
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                :param in_channels: Number of input channels
                :param out_channels: Number of output channels
                :param stride: Stride for the first convolutional layer
                :param downsample: Downsample layer for the shortcut connection
                """
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion),
                )
        self.stride = stride
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                :param x: Input tensor, shape (batch_size, in_channels, height, width)
                :return: Output tensor, shape (batch_size, out_channels, height, width)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        identity = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        out = self.conv1(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        out = self.bn1(out)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        out = self.relu(out)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        out = self.conv2(out)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        out = self.bn2(out)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        if self.downsample is not None:
                    identity = self.downsample(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        if (out.is_cuda and identity.is_cuda and
                out.dtype == torch.float32 and identity.dtype == torch.float32 and
                out.is_contiguous() and identity.is_contiguous() and
                out.shape == identity.shape):
            out = _stark_get_extension().add_relu(out, identity)
        else:
            out = self.relu(out + identity)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_10>>>
        out = out
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_11>>>
        return out
        # <<<END_IMPROVE>>>
