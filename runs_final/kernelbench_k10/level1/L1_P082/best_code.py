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
    return f'stark_cuda_l1_p82_{digest}'

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
#include <c10/util/Optional.h>

torch::Tensor depthwise_conv3x3_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_conv3x3_cuda", &depthwise_conv3x3_cuda,
          "Depthwise 3x3 conv (CUDA, stride=1, padding=0)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

// Logical output tile: TILE_W wide, TILE_H tall
// Each thread computes 2 adjacent outputs in x, so blockDim.x = TILE_W/2
#define TILE_W 32
#define TILE_H 8
#define BLOCK_X (TILE_W / 2)  // 16
#define BLOCK_Y TILE_H         // 8

__global__ void depthwise_conv3x3_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    bool has_bias)
{
    // grid: (ceil(W_out/TILE_W), ceil(H_out/TILE_H), N*C)
    const int nc = blockIdx.z;
    const int n  = nc / C;
    const int c  = nc % C;

    const int tile_oh_start = blockIdx.y * TILE_H;
    const int tile_ow_start = blockIdx.x * TILE_W;

    // Each thread handles two output columns: ow0 and ow1 = ow0+1
    const int oh  = tile_oh_start + threadIdx.y;
    const int ow0 = tile_ow_start + threadIdx.x * 2;
    const int ow1 = ow0 + 1;

    // Shared memory for input tile: (TILE_H+2) x (TILE_W+2)
    __shared__ float smem[(TILE_H + 2)][(TILE_W + 2)];
    __shared__ float sw[9];  // 3x3 weight cache for this channel

    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
    const int block_threads = BLOCK_Y * BLOCK_X;  // 128

    // Load 3x3 weights for this channel into shared memory (first 9 threads)
    if (tid < 9) {
        sw[tid] = __ldg(weight + c * 9 + tid);
    }

    // Load input tile cooperatively
    const int smem_h = TILE_H + 2;
    const int smem_w = TILE_W + 2;
    const int total_smem = smem_h * smem_w;

    const float* in_ptr = input + (n * C + c) * H * W;

    for (int idx = tid; idx < total_smem; idx += block_threads) {
        int si = idx / smem_w;
        int sj = idx % smem_w;
        int ih = tile_oh_start + si;
        int iw = tile_ow_start + sj;
        float val = 0.0f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            val = __ldg(in_ptr + ih * W + iw);
        }
        smem[si][sj] = val;
    }

    __syncthreads();

    const int sy = threadIdx.y;
    const int sx = threadIdx.x * 2;  // base shared-mem x index for ow0

    float bias_val = (has_bias) ? __ldg(bias + c) : 0.0f;

    // Compute output for ow0
    if (oh < H_out && ow0 < W_out) {
        float acc0 = 0.0f;
        acc0 += smem[sy+0][sx+0] * sw[0];
        acc0 += smem[sy+0][sx+1] * sw[1];
        acc0 += smem[sy+0][sx+2] * sw[2];
        acc0 += smem[sy+1][sx+0] * sw[3];
        acc0 += smem[sy+1][sx+1] * sw[4];
        acc0 += smem[sy+1][sx+2] * sw[5];
        acc0 += smem[sy+2][sx+0] * sw[6];
        acc0 += smem[sy+2][sx+1] * sw[7];
        acc0 += smem[sy+2][sx+2] * sw[8];
        if (has_bias) acc0 += bias_val;
        output[(n * C + c) * H_out * W_out + oh * W_out + ow0] = acc0;
    }

    // Compute output for ow1 (reuses smem[sy+*][sx+1..sx+3])
    if (oh < H_out && ow1 < W_out) {
        float acc1 = 0.0f;
        acc1 += smem[sy+0][sx+1] * sw[0];
        acc1 += smem[sy+0][sx+2] * sw[1];
        acc1 += smem[sy+0][sx+3] * sw[2];
        acc1 += smem[sy+1][sx+1] * sw[3];
        acc1 += smem[sy+1][sx+2] * sw[4];
        acc1 += smem[sy+1][sx+3] * sw[5];
        acc1 += smem[sy+2][sx+1] * sw[6];
        acc1 += smem[sy+2][sx+2] * sw[7];
        acc1 += smem[sy+2][sx+3] * sw[8];
        if (has_bias) acc1 += bias_val;
        output[(n * C + c) * H_out * W_out + oh * W_out + ow1] = acc1;
    }
}

torch::Tensor depthwise_conv3x3_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // stride=1, padding=0, kernel=3
    int H_out = H - 2;
    int W_out = W - 2;

    TORCH_CHECK(H_out > 0 && W_out > 0, "output spatial dims must be positive");

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    auto weight_c = weight.contiguous();

    const float* bias_ptr = nullptr;
    bool has_bias = bias.has_value() && bias.value().defined();
    torch::Tensor bias_c;
    if (has_bias) {
        bias_c = bias.value().contiguous();
        bias_ptr = bias_c.data_ptr<float>();
    }

    dim3 block(BLOCK_X, BLOCK_Y);  // 16 x 8 = 128 threads
    dim3 grid(
        (W_out + TILE_W - 1) / TILE_W,
        (H_out + TILE_H - 1) / TILE_H,
        N * C
    );

    depthwise_conv3x3_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight_c.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        has_bias
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a depthwise 2D convolution operation with square input and square kernel.

        Args:
            in_channels (int): Number of channels in the input tensor.
            kernel_size (int): Size of the convolution kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to the input. Defaults to 0.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the depthwise 2D convolution.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        conv2d = self.conv2d
        if (x.is_cuda and
            x.dtype == torch.float32 and
            x.dim() == 4 and
            x.is_contiguous() and
            conv2d.groups == conv2d.in_channels and
            conv2d.in_channels == conv2d.out_channels and
            conv2d.kernel_size == (3, 3) and
            conv2d.stride == (1, 1) and
            conv2d.padding == (0, 0) and
            conv2d.dilation == (1, 1)):
            return _stark_get_extension().depthwise_conv3x3_cuda(
                x, conv2d.weight, conv2d.bias)
        return conv2d(x)
        # <<<END_IMPROVE>>>
