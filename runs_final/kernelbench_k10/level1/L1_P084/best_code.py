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
    return f'stark_cuda_l1_p84_{digest}'

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

torch::Tensor depthwise_conv3x3_s1_p0_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_conv3x3_s1_p0", &depthwise_conv3x3_s1_p0_cuda,
          "Depthwise 3x3 conv stride=1 padding=0 (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 16x32 output tile computed by a 16x16 thread block (256 threads/block).
// Each thread computes 2 adjacent output columns, improving occupancy vs 512-thread block.
#define TILE_H 16
#define TILE_W 32
#define THREAD_H 16
#define THREAD_W 16
// Shared memory holds the input halo: (TILE_H+2) x (TILE_W+2), +1 col to avoid bank conflicts.
#define SMEM_W (TILE_W + 2 + 1)
#define SMEM_H (TILE_H + 2)

__global__ void __launch_bounds__(THREAD_H * THREAD_W, 2)
depthwise_conv3x3_s1_p0_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    bool has_bias)
{
    // Block maps to one (n, c) slice and one TILE_H x TILE_W spatial output tile.
    const int nc       = blockIdx.z;
    const int n        = nc / C;
    const int c        = nc % C;
    const int tile_col = blockIdx.x * TILE_W;  // top-left output col of this tile
    const int tile_row = blockIdx.y * TILE_H;  // top-left output row of this tile

    const int tx = threadIdx.x;  // [0, THREAD_W)
    const int ty = threadIdx.y;  // [0, THREAD_H)

    // --- Load 9 kernel weights into registers ---
    // weight layout: [C, 1, 3, 3] -> channel c starts at c*9
    const float* w = weight + c * 9;
    const float w00 = w[0], w01 = w[1], w02 = w[2];
    const float w10 = w[3], w11 = w[4], w12 = w[5];
    const float w20 = w[6], w21 = w[7], w22 = w[8];

    // --- Collaboratively load input halo into shared memory ---
    __shared__ float smem[SMEM_H][SMEM_W];

    const float* in_ptr = input + (n * C + c) * H_in * W_in;
    const int in_row0 = tile_row;  // padding=0, stride=1
    const int in_col0 = tile_col;

    const int num_threads = THREAD_H * THREAD_W;
    const int tid = ty * THREAD_W + tx;
    // Load SMEM_H x (TILE_W+2) elements (the actual halo, ignoring the +1 padding col)
    const int smem_elems = SMEM_H * (TILE_W + 2);

    for (int idx = tid; idx < smem_elems; idx += num_threads) {
        int sr = idx / (TILE_W + 2);
        int sc = idx % (TILE_W + 2);
        int gr = in_row0 + sr;
        int gc = in_col0 + sc;
        float val = 0.0f;
        if (gr >= 0 && gr < H_in && gc >= 0 && gc < W_in) {
            val = __ldg(in_ptr + gr * W_in + gc);
        }
        smem[sr][sc] = val;
    }

    __syncthreads();

    // --- Each thread computes two adjacent output columns ---
    // Thread (tx, ty) handles out_col0 = tile_col + tx*2  and  out_col1 = tile_col + tx*2 + 1
    const int out_row  = tile_row + ty;
    const int out_col0 = tile_col + tx * 2;
    const int out_col1 = out_col0 + 1;

    // smem column indices for the two windows
    const int sx0 = tx * 2;   // smem col for first window start
    const int sx1 = sx0 + 1;  // smem col for second window start

    if (out_row < H_out) {
        // First output column
        if (out_col0 < W_out) {
            float sum0 =
                w00 * smem[ty  ][sx0  ] + w01 * smem[ty  ][sx0+1] + w02 * smem[ty  ][sx0+2] +
                w10 * smem[ty+1][sx0  ] + w11 * smem[ty+1][sx0+1] + w12 * smem[ty+1][sx0+2] +
                w20 * smem[ty+2][sx0  ] + w21 * smem[ty+2][sx0+1] + w22 * smem[ty+2][sx0+2];
            if (has_bias) sum0 += bias[c];
            output[(n * C + c) * H_out * W_out + out_row * W_out + out_col0] = sum0;
        }
        // Second output column
        if (out_col1 < W_out) {
            float sum1 =
                w00 * smem[ty  ][sx1  ] + w01 * smem[ty  ][sx1+1] + w02 * smem[ty  ][sx1+2] +
                w10 * smem[ty+1][sx1  ] + w11 * smem[ty+1][sx1+1] + w12 * smem[ty+1][sx1+2] +
                w20 * smem[ty+2][sx1  ] + w21 * smem[ty+2][sx1+1] + w22 * smem[ty+2][sx1+2];
            if (has_bias) sum1 += bias[c];
            output[(n * C + c) * H_out * W_out + out_row * W_out + out_col1] = sum1;
        }
    }
}

torch::Tensor depthwise_conv3x3_s1_p0_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "weight must be contiguous CUDA tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");

    const int N   = x.size(0);
    const int C   = x.size(1);
    const int H   = x.size(2);
    const int W   = x.size(3);
    const int H_out = H - 2;  // stride=1, padding=0, kernel=3
    const int W_out = W - 2;

    TORCH_CHECK(H_out > 0 && W_out > 0, "Input spatial dims too small for 3x3 kernel with no padding");

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    bool has_bias = (bias.numel() > 0);

    // Each thread covers 2 output columns, so grid.x based on TILE_W (32 cols per block)
    // block is THREAD_W x THREAD_H = 16x16 = 256 threads
    dim3 block(THREAD_W, THREAD_H);
    dim3 grid(
        (W_out + TILE_W - 1) / TILE_W,
        (H_out + TILE_H - 1) / TILE_H,
        N * C
    );

    depthwise_conv3x3_s1_p0_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, H_out, W_out,
        has_bias
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a depthwise 2D convolution with asymmetric input and square kernel.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the square convolution kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to the input. Defaults to 0.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the depthwise 2D convolution.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        conv2d = self.conv2d
        if (x.is_cuda and
                x.dtype == torch.float32 and
                x.is_contiguous() and
                conv2d.groups == conv2d.in_channels and
                conv2d.out_channels == conv2d.in_channels and
                conv2d.kernel_size == (3, 3) and
                conv2d.stride == (1, 1) and
                conv2d.padding == (0, 0) and
                conv2d.dilation == (1, 1)):
            bias_tensor = conv2d.bias if conv2d.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            return _stark_get_extension().depthwise_conv3x3_s1_p0(x, conv2d.weight, bias_tensor)
        return conv2d(x)
        # <<<END_IMPROVE>>>
