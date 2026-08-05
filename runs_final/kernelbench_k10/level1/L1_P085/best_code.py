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
    return f'stark_cuda_l1_p85_{digest}'

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

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_conv2d_cuda", &depthwise_conv2d_cuda,
          "Depthwise 2D convolution (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

#define TILE_W 32
#define TILE_H 4
#define MAX_WEIGHT_ELEMS 8192

__constant__ float c_weight[MAX_WEIGHT_ELEMS];

__global__ void __launch_bounds__(TILE_W * TILE_H, 4)
depthwise_conv2d_kernel_3x7(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    bool has_bias,
    int N, int C,
    int in_h, int in_w,
    int out_h, int out_w
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_col_base = blockIdx.x * TILE_W;
    const int out_row_base = blockIdx.y * TILE_H;
    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc % C;

    __shared__ float smem[(TILE_H + 2) * (TILE_W + 6)];

    const int smem_h = TILE_H + 2;
    const int smem_w = TILE_W + 6;
    const int in_row0 = out_row_base - 0;
    const int in_col0 = out_col_base - 0;
    const float* in_ptr = input + (n * C + c) * in_h * in_w;

    for (int sr = ty; sr < smem_h; sr += TILE_H) {
        for (int sc = tx; sc < smem_w; sc += TILE_W) {
            int ir = in_row0 + sr;
            int ic = in_col0 + sc;
            float val = 0.0f;
            if ((unsigned)ir < (unsigned)in_h && (unsigned)ic < (unsigned)in_w) {
                val = in_ptr[ir * in_w + ic];
            }
            smem[sr * smem_w + sc] = val;
        }
    }

    __syncthreads();

    const int out_row = out_row_base + ty;
    const int out_col = out_col_base + tx;
    if (out_row >= out_h || out_col >= out_w) return;

    const int w_offset = c * 21;
    const int smem_row_start = ty;
    const int smem_col_start = tx;

    float acc = 0.0f;
    acc += smem[(smem_row_start + 0) * smem_w + (smem_col_start + 0)] * c_weight[w_offset + 0];
    acc += smem[(smem_row_start + 0) * smem_w + (smem_col_start + 1)] * c_weight[w_offset + 1];
    acc += smem[(smem_row_start + 0) * smem_w + (smem_col_start + 2)] * c_weight[w_offset + 2];
    acc += smem[(smem_row_start + 0) * smem_w + (smem_col_start + 3)] * c_weight[w_offset + 3];
    acc += smem[(smem_row_start + 0) * smem_w + (smem_col_start + 4)] * c_weight[w_offset + 4];
    acc += smem[(smem_row_start + 0) * smem_w + (smem_col_start + 5)] * c_weight[w_offset + 5];
    acc += smem[(smem_row_start + 0) * smem_w + (smem_col_start + 6)] * c_weight[w_offset + 6];
    acc += smem[(smem_row_start + 1) * smem_w + (smem_col_start + 0)] * c_weight[w_offset + 7];
    acc += smem[(smem_row_start + 1) * smem_w + (smem_col_start + 1)] * c_weight[w_offset + 8];
    acc += smem[(smem_row_start + 1) * smem_w + (smem_col_start + 2)] * c_weight[w_offset + 9];
    acc += smem[(smem_row_start + 1) * smem_w + (smem_col_start + 3)] * c_weight[w_offset + 10];
    acc += smem[(smem_row_start + 1) * smem_w + (smem_col_start + 4)] * c_weight[w_offset + 11];
    acc += smem[(smem_row_start + 1) * smem_w + (smem_col_start + 5)] * c_weight[w_offset + 12];
    acc += smem[(smem_row_start + 1) * smem_w + (smem_col_start + 6)] * c_weight[w_offset + 13];
    acc += smem[(smem_row_start + 2) * smem_w + (smem_col_start + 0)] * c_weight[w_offset + 14];
    acc += smem[(smem_row_start + 2) * smem_w + (smem_col_start + 1)] * c_weight[w_offset + 15];
    acc += smem[(smem_row_start + 2) * smem_w + (smem_col_start + 2)] * c_weight[w_offset + 16];
    acc += smem[(smem_row_start + 2) * smem_w + (smem_col_start + 3)] * c_weight[w_offset + 17];
    acc += smem[(smem_row_start + 2) * smem_w + (smem_col_start + 4)] * c_weight[w_offset + 18];
    acc += smem[(smem_row_start + 2) * smem_w + (smem_col_start + 5)] * c_weight[w_offset + 19];
    acc += smem[(smem_row_start + 2) * smem_w + (smem_col_start + 6)] * c_weight[w_offset + 20];

    if (has_bias) acc += bias[c];
    output[(n * C + c) * out_h * out_w + out_row * out_w + out_col] = acc;
}

__global__ void __launch_bounds__(TILE_W * TILE_H, 4)
depthwise_conv2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    bool has_bias,
    int N, int C,
    int in_h, int in_w,
    int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_col_base = blockIdx.x * TILE_W;
    const int out_row_base = blockIdx.y * TILE_H;
    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc % C;

    extern __shared__ float smem[];

    const int smem_h = (TILE_H - 1) * stride_h + (kernel_h - 1) * dilation_h + 1;
    const int smem_w = (TILE_W - 1) * stride_w + (kernel_w - 1) * dilation_w + 1;
    const int in_row0 = out_row_base * stride_h - padding_h;
    const int in_col0 = out_col_base * stride_w - padding_w;
    const float* in_ptr = input + (n * C + c) * in_h * in_w;

    for (int sr = ty; sr < smem_h; sr += TILE_H) {
        for (int sc = tx; sc < smem_w; sc += TILE_W) {
            int ir = in_row0 + sr;
            int ic = in_col0 + sc;
            float val = 0.0f;
            if ((unsigned)ir < (unsigned)in_h && (unsigned)ic < (unsigned)in_w) {
                val = in_ptr[ir * in_w + ic];
            }
            smem[sr * smem_w + sc] = val;
        }
    }

    __syncthreads();

    const int out_row = out_row_base + ty;
    const int out_col = out_col_base + tx;
    if (out_row >= out_h || out_col >= out_w) return;

    const int w_offset = c * kernel_h * kernel_w;
    float acc = 0.0f;
    const int smem_row_start = ty * stride_h;
    const int smem_col_start = tx * stride_w;

    #pragma unroll 1
    for (int kh = 0; kh < kernel_h; kh++) {
        int sr = smem_row_start + kh * dilation_h;
        #pragma unroll 1
        for (int kw = 0; kw < kernel_w; kw++) {
            int sc = smem_col_start + kw * dilation_w;
            acc += smem[sr * smem_w + sc] * c_weight[w_offset + kh * kernel_w + kw];
        }
    }

    if (has_bias) acc += bias[c];
    output[(n * C + c) * out_h * out_w + out_row * out_w + out_col] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.size(1) == 1, "weight must have size(1)==1 for depthwise conv");

    const int N = input.size(0);
    const int C = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    int num_weight_elems = C * kernel_h * kernel_w;
    TORCH_CHECK(num_weight_elems <= MAX_WEIGHT_ELEMS, "Too many weight elements for constant memory");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(),
                       num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    auto output = torch::empty({N, C, out_h, out_w}, input.options());

    const float* bias_ptr = nullptr;
    bool has_bias = false;
    if (bias.has_value() && bias.value().defined()) {
        has_bias = true;
        bias_ptr = bias.value().contiguous().data_ptr<float>();
    }

    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_w + TILE_W - 1) / TILE_W,
        (out_h + TILE_H - 1) / TILE_H,
        N * C
    );

    bool use_specialized = (kernel_h == 3 && kernel_w == 7 &&
                            stride_h == 1 && stride_w == 1 &&
                            padding_h == 0 && padding_w == 0 &&
                            dilation_h == 1 && dilation_w == 1);

    if (use_specialized) {
        int smem_bytes = (TILE_H + 2) * (TILE_W + 6) * sizeof(float);
        depthwise_conv2d_kernel_3x7<<<grid, block, smem_bytes>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            bias_ptr,
            has_bias,
            N, C,
            in_h, in_w,
            out_h, out_w
        );
    } else {
        int smem_h = (TILE_H - 1) * stride_h + (kernel_h - 1) * dilation_h + 1;
        int smem_w = (TILE_W - 1) * stride_w + (kernel_w - 1) * dilation_w + 1;
        int smem_bytes = smem_h * smem_w * sizeof(float);
        depthwise_conv2d_kernel<<<grid, block, smem_bytes>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            bias_ptr,
            has_bias,
            N, C,
            in_h, in_w,
            out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size_h (int): Height of the convolution kernel.
            kernel_size_w (int): Width of the convolution kernel.
            stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
            stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
            padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
            padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
            dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
            dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv2d = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias)
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
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            weight = self.conv2d.weight.contiguous()
            bias_tensor = self.conv2d.bias.contiguous() if self.conv2d.bias is not None else None
            return _stark_get_extension().depthwise_conv2d_cuda(
            x, weight, bias_tensor,
            self.conv2d.stride[0], self.conv2d.stride[1],
            self.conv2d.padding[0], self.conv2d.padding[1],
            self.conv2d.dilation[0], self.conv2d.dilation[1]
            )
        return self.conv2d(x)
        # <<<END_IMPROVE>>>
