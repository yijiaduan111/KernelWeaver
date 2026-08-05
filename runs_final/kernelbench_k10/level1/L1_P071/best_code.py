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
    return f'stark_cuda_l1_p71_{digest}'

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

torch::Tensor conv_transpose2d_direct_forward(torch::Tensor input, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_direct_forward", &conv_transpose2d_direct_forward,
          "Output-stationary direct gather ConvTranspose2d k3 s1 (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CTDCONV_BW 32
#define CTDCONV_BH 8

// Output-stationary direct gather kernel for 3x3 stride-1 transposed convolution.
// Each thread computes exactly one output element: no atomics, no intermediate buffers.
// weight layout: [Cin, Cout,3, 3]  (PyTorch ConvTranspose2d convention)
__global__ void __launch_bounds__(256, 4) conv_transpose2d_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int Cin, int Cout,
    int H_in, int W_in,
    int H_out, int W_out
) {
    int ow= blockIdx.x * CTDCONV_BW + threadIdx.x;
    int oh   = blockIdx.y * CTDCONV_BH + threadIdx.y;
    int co_n = blockIdx.z;
    int co   = co_n % Cout;
    int n    = co_n / Cout;

    if (ow >= W_out || oh >= H_out) return;

    float acc = 0.0f;

    // Outer loop over input channels; NOT unrolled to keep register pressure low.
    for (int ci = 0; ci < Cin; ci++) {
        const float* inp_ptr = input  + (n * Cin + ci) * H_in * W_in;
        const float* wt_ptr  = weight + (ci * Cout + co) * 9;

        // Unroll only the 3x3 kernel loops.
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            int ih = oh - kh;  // stride=1, padding=0
            if (ih < 0 || ih >= H_in) continue;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int iw = ow - kw;
                if (iw < 0 || iw >= W_in) continue;
                acc += __ldg(inp_ptr + ih * W_in + iw) * __ldg(wt_ptr + kh * 3 + kw);
            }
        }
    }

    output[((n * Cout + co) * H_out + oh) * W_out + ow] = acc;
}

torch::Tensor conv_transpose2d_direct_forward(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(),"input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim()  == 4, "input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3, "kernel must be 3x3");

    const int N    = input.size(0);
    const int Cin  = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int Cout = weight.size(1);
    // stride=1, padding=0, k=3: H_out = H_in + k - 1
    const int H_out = H_in + 2;
    const int W_out = W_in + 2;

    TORCH_CHECK(weight.size(0) == Cin, "weight input channels must match input channels");

    auto output = torch::empty({N, Cout, H_out, W_out}, input.options());

    dim3 block(CTDCONV_BW, CTDCONV_BH, 1);
    dim3 grid(
        (W_out + CTDCONV_BW - 1) / CTDCONV_BW,
        (H_out + CTDCONV_BH - 1) / CTDCONV_BH,
        N * Cout
    );

    conv_transpose2d_direct_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, Cout,
        H_in, W_in,
        H_out, W_out
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a transposed 2D convolution with asymmetric input and a square kernel.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the square convolution kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to the input. Defaults to 0.
            output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the transposed 2D convolution.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return self.conv_transpose2d(x)
        # <<<END_IMPROVE>>>
