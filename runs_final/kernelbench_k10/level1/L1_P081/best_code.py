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
    return f'stark_cuda_l1_p81_{digest}'

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

torch::Tensor conv_transpose2d_s5_p1_d2_cuda(torch::Tensor input, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_s5_p1_d2", &conv_transpose2d_s5_p1_d2_cuda, "Specialized ConvTranspose2d k3 s5 p1 d2 (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Specialized kernel: ConvTranspose2d, kernel=3x3, stride=5, padding=1, dilation=2, no bias
// Weight layout: [Cin, Cout, kH, kW] (PyTorch ConvTranspose2d convention)
// For this regime: P=1, D=2, S=5
// Valid taps per output dim: (o + P - k*D) must be >= 0 and divisible by S
// For kh in {0,1,2}, D=2, P=1, S=5:
//   kh=0: need (oh+1) % 5 == 0  => oh mod 5 == 4
//   kh=1: need (oh-1) % 5 == 0  => oh mod 5 == 1
//   kh=2: need (oh-3) % 5 == 0  => oh mod 5 == 3
// So each output row is valid for at most one kh. Same for kw.

#define CONV_S 5
#define CONV_P 1
#define CONV_D 2
#define CONV_K 3

// Map residue -> kh/kw index, returns -1 if invalid
__device__ __forceinline__ int residue_to_k(int residue) {
    if (residue == 0) return 0;
    if (residue == 2) return 1;
    if (residue == 4) return 2;
    return -1;
}

__global__ void __launch_bounds__(256, 6)
conv_transpose2d_k3s5p1d2_kernel(
    const float* __restrict__ input,   // [N, Cin, IH, IW]
    const float* __restrict__ weight,  // [Cin, Cout, K, K]
    float* __restrict__ output,        // [N, Cout, OH, OW]
    int N, int Cin, int Cout,
    int IH, int IW,
    int OH, int OW
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * Cout * OH * OW;
    if (idx >= total) return;

    int tmp = idx;
    const int ow = tmp % OW; tmp /= OW;
    const int oh = tmp % OH; tmp /= OH;
    const int oc = tmp % Cout; tmp /= Cout;
    const int n  = tmp;

    // Direct residue-based tap decoding - at most one valid (kh, kw) pair
    const int rh = (oh + CONV_P) % CONV_S;
    const int kh = residue_to_k(rh);
    if (kh < 0) { output[idx] = 0.0f; return; }

    const int rw = (ow + CONV_P) % CONV_S;
    const int kw = residue_to_k(rw);
    if (kw < 0) { output[idx] = 0.0f; return; }

    const int ih = (oh + CONV_P - kh * CONV_D) / CONV_S;
    const int iw = (ow + CONV_P - kw * CONV_D) / CONV_S;

    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) { output[idx] = 0.0f; return; }

    // Weight offset for this (oc, kh, kw): weight[ic, oc, kh, kw]
    const int w_tap_offset = oc * CONV_K * CONV_K + kh * CONV_K + kw;
    const int inp_hw_offset = n * Cin * IH * IW + ih * IW + iw;
    const int w_cin_stride = Cout * CONV_K * CONV_K;
    const int inp_hw_stride = IH * IW;

    const float* in_ptr = input + inp_hw_offset;
    const float* w_ptr  = weight + w_tap_offset;

    float acc = 0.0f;
    if (Cin == 32) {
        #pragma unroll
        for (int ic = 0; ic < 32; ++ic) {
            acc += in_ptr[ic * inp_hw_stride] * w_ptr[ic * w_cin_stride];
        }
    } else {
        for (int ic = 0; ic < Cin; ++ic) {
            acc += in_ptr[ic * inp_hw_stride] * w_ptr[ic * w_cin_stride];
        }
    }

    output[idx] = acc;
}

torch::Tensor conv_transpose2d_s5_p1_d2_cuda(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Tensors must be contiguous");

    const int N    = input.size(0);
    const int Cin  = input.size(1);
    const int IH   = input.size(2);
    const int IW   = input.size(3);
    const int Cout = weight.size(1);

    const int OH = (IH - 1) * CONV_S - 2 * CONV_P + CONV_D * (CONV_K - 1) + 1;
    const int OW = (IW - 1) * CONV_S - 2 * CONV_P + CONV_D * (CONV_K - 1) + 1;

    auto output = torch::empty({N, Cout, OH, OW}, input.options());

    const int total = N * Cout * OH * OW;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    conv_transpose2d_k3s5p1d2_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, Cout, IH, IW, OH, OW
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to the input. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        def _pair(v):
            return (v, v) if isinstance(v, int) else tuple(v)
        self._fast_path = (
        _pair(kernel_size) == (3, 3) and
        _pair(stride) == (5, 5) and
        _pair(padding) == (1, 1) and
        _pair(dilation) == (2, 2) and
        not bias
        )
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the 2D transposed convolution.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (
            self._fast_path
            and x.is_cuda
            and x.dtype == torch.float32
            and x.is_contiguous()
            and self.conv_transpose2d.weight.is_contiguous()
        ):
            return _stark_get_extension().conv_transpose2d_s5_p1_d2(x, self.conv_transpose2d.weight)
        return self.conv_transpose2d(x)
        # <<<END_IMPROVE>>>
