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
    return f'stark_cuda_l1_p78_{digest}'

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

torch::Tensor conv_transpose2d_fast(torch::Tensor x, torch::Tensor weight, torch::optional<torch::Tensor> bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_fast", &conv_transpose2d_fast, "Specialized ConvTranspose2d fastpath");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int H, int W, int Cin, int Cout,
    int KH, int KW, int pad_h, int pad_w,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * H * W;
    if (idx >= total) return;

    int x = idx % W;
    int temp = idx / W;
    int y = temp % H;
    temp = temp / H;
    int co = temp % Cout;
    int n = temp / Cout;

    float sum = has_bias ? bias[co] : 0.0f;

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int in_y = y + pad_h - kh;
                int in_x = x + pad_w - kw;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    float in_val = input[((n * Cin + ci) * H + in_y) * W + in_x];
                    float w_val = weight[((ci * Cout + co) * KH + kh) * KW + kw];
                    sum += in_val * w_val;
                }
            }
        }
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose2d_fast(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(x.size(1) == 32, "x.size(1) must be 32");
    TORCH_CHECK(weight.size(0) == 32, "weight.size(0) must be 32");
    TORCH_CHECK(weight.size(1) == 32, "weight.size(1) must be 32");
    TORCH_CHECK(weight.size(2) == 3, "weight.size(2) must be 3");
    TORCH_CHECK(weight.size(3) == 7, "weight.size(3) must be 7");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(1);
    int KH = weight.size(2);
    int KW = weight.size(3);
    int pad_h = 1;
    int pad_w = 3;

    bool has_bias = bias.has_value();
    if (has_bias) {
        TORCH_CHECK(bias->is_cuda(), "bias must be CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    }

    auto output = torch::empty({N, Cout, H, W}, x.options());

    int total_elements = N * Cout * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, H, W, Cin, Cout, KH, KW, pad_h, pad_w, has_bias
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (tuple): Size of the convolution kernel (height, width).
            stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
            padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the 2D transposed convolution.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return self.conv_transpose2d(x)
        # <<<END_IMPROVE>>>
