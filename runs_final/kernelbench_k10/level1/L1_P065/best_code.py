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
    return f'stark_cuda_l1_p65_{digest}'

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

torch::Tensor conv_transpose2d_fast(torch::Tensor input, torch::Tensor weight, int pad_h, int pad_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_fast", &conv_transpose2d_fast, "specialized conv_transpose2d fast path");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each thread computes 2 horizontally adjacent output elements (out_x0, out_x0+1)
// to amortize weight/input row loads across two accumulators.
__global__ void conv_transpose2d_kernel_3x7_t2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int pad_h, int pad_w
) {
    // Each thread covers 2 output x positions
    int out_x0 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int out_y  = blockIdx.y * blockDim.y + threadIdx.y;
    int n  = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;

    if (out_x0 >= out_w || out_y >= out_h || n >= batch) return;

    int out_x1 = out_x0 + 1;
    bool valid1 = (out_x1 < out_w);

    // Shared ky range (depends only on out_y)
    int ky_begin = max(0, out_y + pad_h - (in_h - 1));
    int ky_end   = min(3, out_y + pad_h + 1);

    // Per-x kx ranges
    int kx0_begin = max(0, out_x0 + pad_w - (in_w - 1));
    int kx0_end   = min(7, out_x0 + pad_w + 1);
    int kx1_begin = max(0, out_x1 + pad_w - (in_w - 1));
    int kx1_end   = min(7, out_x1 + pad_w + 1);

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const int ic_spatial = in_h * in_w;

    for (int ic = 0; ic < in_channels; ++ic) {
        const float* in_ic = input  + (n * in_channels + ic) * ic_spatial;
        const float* w_ic  = weight + (ic * out_channels + oc) * 21; // 3*7=21

        #pragma unroll
        for (int ky = ky_begin; ky < ky_end; ++ky) {
            int in_y = out_y + pad_h - ky;
            const float* in_row = in_ic + in_y * in_w;
            const float* w_row  = w_ic  + ky * 7;

            // Accumulate sum0
            #pragma unroll
            for (int kx = kx0_begin; kx < kx0_end; ++kx) {
                sum0 += in_row[out_x0 + pad_w - kx] * w_row[kx];
            }

            // Accumulate sum1
            if (valid1) {
                #pragma unroll
                for (int kx = kx1_begin; kx < kx1_end; ++kx) {
                    sum1 += in_row[out_x1 + pad_w - kx] * w_row[kx];
                }
            }
        }
    }

    int base_out = ((n * out_channels + oc) * out_h + out_y) * out_w;
    output[base_out + out_x0] = sum0;
    if (valid1) output[base_out + out_x1] = sum1;
}

torch::Tensor conv_transpose2d_fast(torch::Tensor input, torch::Tensor weight, int pad_h, int pad_w) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int batch        = input.size(0);
    int in_channels  = input.size(1);
    int in_h         = input.size(2);
    int in_w         = input.size(3);
    int out_channels = weight.size(1);
    int kh           = weight.size(2);
    int kw           = weight.size(3);

    int out_h = in_h - 1 - 2 * pad_h + kh;
    int out_w = in_w - 1 - 2 * pad_w + kw;

    auto output = torch::zeros({batch, out_channels, out_h, out_w}, input.options());

    // block.x covers 2 output x positions per thread
    dim3 block(16, 8);
    dim3 grid(
        (out_w + block.x * 2 - 1) / (block.x * 2),
        (out_h + block.y - 1) / block.y,
        batch * out_channels
    );

    conv_transpose2d_kernel_3x7_t2<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        pad_h, pad_w
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a transposed 2D convolution with a square input and an asymmetric kernel.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (tuple): Size of the convolution kernel (height, width).
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
            output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )
        if torch.cuda.is_available():
            self.conv_transpose2d = self.conv_transpose2d.to(memory_format=torch.channels_last)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the transposed 2D convolution.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and self.conv_transpose2d.weight.dtype == torch.float32
            and self.conv_transpose2d.bias is None
            and self.conv_transpose2d.groups == 1
            and self.conv_transpose2d.stride == (1, 1)
            and self.conv_transpose2d.dilation == (1, 1)
            and self.conv_transpose2d.output_padding == (0, 0)
            and tuple(self.conv_transpose2d.kernel_size) == (3, 7)
            and x.is_contiguous()
            and self.conv_transpose2d.weight.is_contiguous()
        ):
            return _stark_get_extension().conv_transpose2d_fast(
                x,
                self.conv_transpose2d.weight,
                self.conv_transpose2d.padding[0],
                self.conv_transpose2d.padding[1],
            )
        return self.conv_transpose2d(x)
        # <<<END_IMPROVE>>>
