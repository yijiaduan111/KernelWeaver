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
    return f'stark_cuda_l1_p45_{digest}'

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

torch::Tensor avg_pool2d_k11s11_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool2d_k11s11_cuda", &avg_pool2d_k11s11_cuda, "Specialized AvgPool2d k11 s11 CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void avg_pool2d_k11s11_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    // Decode indices using int32 after bounds check
    int w_out_i = static_cast<int>(W_out);
    int h_out_i = static_cast<int>(H_out);
    int c_i     = static_cast<int>(C);
    int w_i     = static_cast<int>(W);

    int ow  = static_cast<int>(idx % w_out_i);
    int tmp = static_cast<int>(idx / w_out_i);
    int oh  = tmp % h_out_i;
    tmp     = tmp / h_out_i;
    int c   = tmp % c_i;
    int n   = tmp / c_i;

    int ih_base = oh * 11;
    int iw_base = ow * 11;

    // 64-bit base pointer to avoid overflow on large tensors
    int64_t nc = static_cast<int64_t>(n) * C + c;
    const float* in_ptr = input + nc * (H * W) + static_cast<int64_t>(ih_base) * w_i + iw_base;

    float sum = 0.0f;
    #pragma unroll
    for (int kh = 0; kh < 11; kh++) {
        const float* row = in_ptr + kh * w_i;
        #pragma unroll
        for (int kw = 0; kw < 11; kw++) {
            sum += __ldg(row + kw);
        }
    }
    output[idx] = sum * (1.0f / 121.0f);
}

torch::Tensor avg_pool2d_k11s11_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t H_out = (H - 11) / 11 + 1;
    int64_t W_out = (W - 11) / 11 + 1;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    int64_t total = N * C * H_out * W_out;
    int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    TORCH_CHECK(blocks64 <= INT_MAX, "Launch grid too large");
    int blocks = static_cast<int>(blocks64);

    avg_pool2d_k11s11_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, H_out, W_out
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs 2D Average Pooling.
        """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the Average Pooling layer.

                Args:
                    kernel_size (int): Size of the pooling window.
                    stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
                    padding (int, optional): Padding applied to the input tensor. Defaults to 0.
                """
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies 2D Average Pooling to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

                Returns:
                    torch.Tensor: Output tensor with Average Pooling applied.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        ks = self.avg_pool.kernel_size
        ks_val = ks[0] if isinstance(ks, tuple) else ks
        st = self.avg_pool.stride
        st_val = (st[0] if isinstance(st, tuple) else st) if st is not None else ks_val
        pad = self.avg_pool.padding
        pad_val = pad[0] if isinstance(pad, tuple) else pad
        if (
            x.is_cuda and
            x.dtype == torch.float32 and
            x.is_contiguous() and
            x.dim() == 4 and
            ks_val == 11 and
            st_val == 11 and
            pad_val == 0
        ):
            return _stark_get_extension().avg_pool2d_k11s11_cuda(x)
        return self.avg_pool(x)
        # <<<END_IMPROVE>>>
