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
    return f'stark_cuda_l1_p83_{digest}'

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

torch::Tensor depthwise_conv2d_kx1_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    int64_t stride_h,
    int64_t padding_h,
    int64_t dilation_h);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_conv2d_kx1_forward", &depthwise_conv2d_kx1_forward,
          "Depthwise (K x 1) convolution forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Specialized kernel for K=3, stride=1, padding=0, dilation=1
// No boundary checks needed in h (since padding=0 and H_out=H-2)
// w_in == w_out (stride=1, padding=0)
template <int THREADS>
__global__ void depthwise_kx1_k3s1p0d1_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out)
{
    int idx = blockIdx.x * THREADS + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp   = idx / W_out;
    int h_out = tmp % H_out;
    tmp      /= H_out;
    int c     = tmp % C;
    int n     = tmp / C;

    // w_in == w_out, h_in_base == h_out (stride=1, padding=0)
    const float* in_ptr = input  + ((n * C + c) * H + h_out) * W + w_out;
    const float* w_ptr  = weight + c * 3;

    float acc = in_ptr[0]           * w_ptr[0]
              + in_ptr[W]           * w_ptr[1]
              + in_ptr[W + W]       * w_ptr[2];

    if (bias != nullptr) acc += bias[c];
    output[idx] = acc;
}

// Generic kernel (all parameters)
template <int THREADS>
__global__ void depthwise_kx1_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int stride, int padding, int dilation,
    int H_out, int W_out)
{
    int idx = blockIdx.x * THREADS + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp   = idx / W_out;
    int h_out = tmp % H_out;
    tmp      /= H_out;
    int c     = tmp % C;
    int n     = tmp / C;

    int h_in_base = h_out * stride - padding;
    int w_in      = w_out * stride - padding;

    const float* in_ptr = input  + (n * C + c) * H * W;
    const float* w_ptr  = weight + c * K;

    float acc = 0.0f;
    if (w_in >= 0 && w_in < W) {
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            int h_in = h_in_base + k * dilation;
            if (h_in >= 0 && h_in < H) {
                acc += in_ptr[h_in * W + w_in] * w_ptr[k];
            }
        }
    }
    if (bias != nullptr) acc += bias[c];
    output[idx] = acc;
}

torch::Tensor depthwise_conv2d_kx1_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    int64_t stride_h,
    int64_t padding_h,
    int64_t dilation_h)
{
    TORCH_CHECK(input.is_cuda(),  "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim()  == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.size(1) == 1 && weight.size(3) == 1,
                "weight must have shape [C, 1, K, 1]");
    TORCH_CHECK(input.size(1) == weight.size(0),
                "channel mismatch between input and weight");

    int N       = static_cast<int>(input.size(0));
    int C       = static_cast<int>(input.size(1));
    int H       = static_cast<int>(input.size(2));
    int W       = static_cast<int>(input.size(3));
    int K       = static_cast<int>(weight.size(2));
    int stride  = static_cast<int>(stride_h);
    int padding = static_cast<int>(padding_h);
    int dilation= static_cast<int>(dilation_h);

    int H_out = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - 1) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.numel() == C, "bias must have one value per channel");
        bias_ptr = bias.data_ptr<float>();
    }

    constexpr int THREADS = 256;
    int total  = N * C * H_out * W_out;
    int blocks = (total + THREADS - 1) / THREADS;

    if (K == 3 && stride == 1 && padding == 0 && dilation == 1) {
        depthwise_kx1_k3s1p0d1_kernel<THREADS><<<blocks, THREADS>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            N, C, H, W, H_out, W_out);
    } else {
        depthwise_kx1_kernel<THREADS><<<blocks, THREADS>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            N, C, H, W,
            K, stride, padding, dilation,
            H_out, W_out);
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

        Args:
            in_channels (int): Number of channels in the input tensor.
            kernel_size (int): Size of the convolution kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to the input. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
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
        conv = self.conv2d
        if (
        x.is_cuda and
        x.dtype == torch.float32 and
        x.is_contiguous() and
        conv.groups == conv.in_channels and
        conv.kernel_size[1] == 1 and
        conv.weight.is_contiguous()
        ):
            return _stark_get_extension().depthwise_conv2d_kx1_forward(
            x,
            conv.weight,
            conv.bias,
            conv.stride[0],
            conv.padding[0],
            conv.dilation[0],
            )
        return self.conv2d(x)
        # <<<END_IMPROVE>>>
