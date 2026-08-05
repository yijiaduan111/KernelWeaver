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
    return f'stark_cuda_l1_p69_{digest}'

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

torch::Tensor conv_transpose2d_fastpath(torch::Tensor input, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_fastpath", &conv_transpose2d_fastpath, "Specialized ConvTranspose2d fast path");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Transposed conv2d for stride=1, pad=0, dilation=1, groups=1
// output[n, oc, oh, ow] = sum_{ic,kh,kw} input[n,ic,oh-kh,ow-kw] * weight[ic,oc,kh,kw]
// where oh in [0, H+Kh-2], ow in [0, W+Kw-2]
// weight layout: [Cin, Cout, Kh, Kw]
//
// Block: (32 threads for ow-tile, 4 threads for oc-tile) = 128 threads
// Grid: (ceil(Wo/32), ceil(Ho/1), N*ceil(Cout/4))

#define BLOCK_OW 32
#define BLOCK_OC 4

__global__ void conv_transpose2d_k3x5(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int Cin, int H, int W,
    int Cout, int Ho, int Wo)
{
    // Thread indices
    int tw = threadIdx.x; // 0..BLOCK_OW-1
    int toc = threadIdx.y; // 0..BLOCK_OC-1

    int ow_base = blockIdx.x * BLOCK_OW;
    int oh      = blockIdx.y;
    int noc_blk = blockIdx.z; // encodes n and oc block
    int oc_blk  = noc_blk % ((Cout + BLOCK_OC - 1) / BLOCK_OC);
    int n       = noc_blk / ((Cout + BLOCK_OC - 1) / BLOCK_OC);

    int oc = oc_blk * BLOCK_OC + toc;
    int ow = ow_base + tw;

    if (n >= N || oh >= Ho || oc >= Cout || ow >= Wo) return;

    const int KH = 3;
    const int KW = 5;

    float acc = 0.0f;

    // weight[ic, oc, kh, kw] -> index: ((ic*Cout + oc)*KH + kh)*KW + kw
    for (int ic = 0; ic < Cin; ic++) {
        const float* inp_ic = input + (n * Cin + ic) * H * W;
        const float* w_ic   = weight + (ic * Cout + oc) * KH * KW;

        #pragma unroll
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh - kh;
            if (ih < 0 || ih >= H) continue;
            const float* inp_row = inp_ic + ih * W;

            #pragma unroll
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow - kw;
                if (iw >= 0 && iw < W) {
                    acc += __ldg(inp_row + iw) * __ldg(w_ic + kh * KW + kw);
                }
            }
        }
    }

    output[((n * Cout + oc) * Ho + oh) * Wo + ow] = acc;
}

torch::Tensor conv_transpose2d_fastpath(
    torch::Tensor input,
    torch::Tensor weight)
{
    const int N   = input.size(0);
    const int Cin = input.size(1);
    const int H   = input.size(2);
    const int W   = input.size(3);

    const int Cout = weight.size(1);
    const int kh   = weight.size(2);
    const int kw   = weight.size(3);

    const int Ho = H + kh - 1;
    const int Wo = W + kw - 1;

    auto output = torch::zeros({N, Cout, Ho, Wo}, input.options());

    int oc_blocks = (Cout + BLOCK_OC - 1) / BLOCK_OC;
    dim3 block(BLOCK_OW, BLOCK_OC);
    dim3 grid((Wo + BLOCK_OW - 1) / BLOCK_OW,
              Ho,
              N * oc_blocks);

    conv_transpose2d_k3x5<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, H, W, Cout, Ho, Wo
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a transposed 2D convolution operation with asymmetric input and kernel size.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
            stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
            padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
            output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
            dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        import os as _os
        self._stark_enable_custom = (_os.environ.get("STARK_USE_CUSTOM_CONVTRANSPOSE2D", "0") == "1")
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        weight = self.conv_transpose2d.weight
        if (
        self._stark_enable_custom
        and x.is_cuda and x.is_contiguous() and x.dtype == torch.float32
        and self.conv_transpose2d.stride == (1, 1)
        and self.conv_transpose2d.padding == (0, 0)
        and self.conv_transpose2d.output_padding == (0, 0)
        and self.conv_transpose2d.dilation == (1, 1)
        and self.conv_transpose2d.groups == 1
        and self.conv_transpose2d.bias is None
        and self.conv_transpose2d.kernel_size == (3, 5)
        and weight.is_contiguous()
        ):
            return _stark_get_extension().conv_transpose2d_fastpath(x, weight)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return self.conv_transpose2d(x)
        # <<<END_IMPROVE>>>
