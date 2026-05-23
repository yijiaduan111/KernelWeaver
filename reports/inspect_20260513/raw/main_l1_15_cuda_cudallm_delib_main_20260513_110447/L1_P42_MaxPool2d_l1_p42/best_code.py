import hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# <<<IMPROVE:helpers>>>
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
    return f'stark_cuda_l1_p42_{digest}'

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
# <<<END_IMPROVE>>>

CUDA_CPP_SRC = r'''
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>

torch::Tensor max_pool2d_forward_cuda(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation);

// Add pybind exports for custom CUDA entrypoints here.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_forward_cuda", &max_pool2d_forward_cuda, "MaxPool2d forward (CUDA)");
}
# <<<END_IMPROVE>>>
'''

CUDA_CU_SRC = r'''
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void max_pool2d_forward_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int kernel_size, int stride, int padding, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_H * out_W;
    if (idx >= total) return;

    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int c = (idx / (out_W * out_H)) % C;
    int n = idx / (out_W * out_H * C);

    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    float max_val = -FLT_MAX;
    int base = ((n * C + c) * H) * W;

    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_start + kh * dilation;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_start + kw * dilation;
            if (w_in < 0 || w_in >= W) continue;
            float val = input[base + h_in * W + w_in];
            max_val = val > max_val ? val : max_val;
        }
    }

    output[idx] = max_val;
}

torch::Tensor max_pool2d_forward_cuda(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "only float32 is supported");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW 4D tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int N = (int)input.size(0);
    int C = (int)input.size(1);
    int H = (int)input.size(2);
    int W = (int)input.size(3);

    int k = (int)kernel_size;
    int s = (int)stride;
    int p = (int)padding;
    int d = (int)dilation;

    int out_H = (H + 2 * p - d * (k - 1) - 1) / s + 1;
    int out_W = (W + 2 * p - d * (k - 1) - 1) / s + 1;

    auto output = torch::empty({N, C, out_H, out_W}, input.options());

    int total = N * C * out_H * out_W;
    int block = 256;
    int grid = (total + block - 1) / block;

    max_pool2d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, out_H, out_W, k, s, p, d
    );

    return output;
}
// Add CUDA kernels and exported wrapper functions here.
# <<<END_IMPROVE>>>
'''

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.is_contiguous()
        ):
            ext = _stark_get_extension()
            k = self.maxpool.kernel_size
            s = self.maxpool.stride
            p = self.maxpool.padding
            d = self.maxpool.dilation
            if isinstance(k, tuple): k = k[0]
            if isinstance(s, tuple): s = s[0]
            if isinstance(p, tuple): p = p[0]
            if isinstance(d, tuple): d = d[0]
            return ext.max_pool2d_forward_cuda(x, int(k), int(s), int(p), int(d))
        return self.maxpool(x)
        # <<<END_IMPROVE>>>