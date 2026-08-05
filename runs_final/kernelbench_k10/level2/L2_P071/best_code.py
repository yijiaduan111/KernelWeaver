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
    return f'stark_cuda_l2_p71_{digest}'

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
#include <ATen/ATen.h>

torch::Tensor conv_bias_divide_leakyrelu_cuda(torch::Tensor y, torch::Tensor bias, double divisor);
torch::Tensor conv_divide_leakyrelu_cuda(torch::Tensor y, double divisor);

torch::Tensor conv_divide_leakyrelu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups,
    double divisor
) {
    if (x.is_cuda() && x.scalar_type() == at::kFloat) {
        // Run conv without bias; fuse bias+divide+leakyrelu in one epilogue pass
        c10::optional<at::Tensor> no_bias;
        torch::Tensor y = at::conv2d(
            x,
            weight,
            no_bias,
            {stride_h, stride_w},
            {pad_h, pad_w},
            {dil_h, dil_w},
            groups
        );
        if (y.is_contiguous()) {
            if (bias.defined() && bias.numel() > 0) {
                return conv_bias_divide_leakyrelu_cuda(y, bias.contiguous(), divisor);
            } else {
                return conv_divide_leakyrelu_cuda(y, divisor);
            }
        }
        // fallback for non-contiguous
        auto divided = (y + bias.view({1, -1, 1, 1})) / divisor;
        return at::leaky_relu(divided, 0.01);
    }
    // CPU fallback
    torch::Tensor y = at::conv2d(
        x,
        weight,
        bias,
        {stride_h, stride_w},
        {pad_h, pad_w},
        {dil_h, dil_w},
        groups
    );
    auto divided = y / divisor;
    return at::leaky_relu(divided, 0.01);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_divide_leakyrelu", &conv_divide_leakyrelu, "Fused conv + bias + divide + leaky_relu");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// In-place: add per-channel bias, multiply by inv_div, apply LeakyReLU
__global__ void bias_divide_leakyrelu_kernel(
    float* __restrict__ data,
    const float* __restrict__ bias,
    float inv_div,
    int64_t spatial,   // H*W
    int64_t C,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t c = (idx / spatial) % C;
        float v = (data[idx] + bias[c]) * inv_div;
        data[idx] = (v >= 0.0f) ? v : (0.01f * v);
    }
}

// In-place: multiply by inv_div, apply LeakyReLU (no bias)
__global__ void divide_leakyrelu_inplace_kernel(
    float* __restrict__ data,
    float inv_div,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = data[idx] * inv_div;
        data[idx] = (v >= 0.0f) ? v : (0.01f * v);
    }
}

torch::Tensor conv_bias_divide_leakyrelu_cuda(torch::Tensor y, torch::Tensor bias, double divisor) {
    int64_t n = y.numel();
    int64_t C = y.size(1);
    int64_t spatial = y.size(2) * y.size(3);
    float inv_div = 1.0f / static_cast<float>(divisor);

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    bias_divide_leakyrelu_kernel<<<blocks, threads>>>(
        y.data_ptr<float>(),
        bias.data_ptr<float>(),
        inv_div,
        spatial,
        C,
        n
    );

    return y;
}

torch::Tensor conv_divide_leakyrelu_cuda(torch::Tensor y, double divisor) {
    int64_t n = y.numel();
    float inv_div = 1.0f / static_cast<float>(divisor);

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    divide_leakyrelu_inplace_kernel<<<blocks, threads>>>(
        y.data_ptr<float>(),
        inv_div,
        n
    );

    return y;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
        """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        if x.is_cuda:
            x = _stark_get_extension().conv_divide_leakyrelu(
            x,
            self.conv.weight,
            self.conv.bias if self.conv.bias is not None else torch.empty(0, device=x.device),
            self.conv.stride[0],
            self.conv.stride[1],
            self.conv.padding[0],
            self.conv.padding[1],
            self.conv.dilation[0],
            self.conv.dilation[1],
            self.conv.groups,
            float(self.divisor)
            )
        else:
            x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if not x.is_cuda:
            x = x / self.divisor
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not x.is_cuda:
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
