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
    return f'stark_cuda_l2_p69_{digest}'

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

torch::Tensor fused_bias_hardswish_relu(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_hardswish_relu", &fused_bias_hardswish_relu, "Fused bias+relu(hardswish(x)) CUDA kernel");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Grid-stride scalar kernel: adds bias[channel] then applies relu(hardswish(v)).
// NCHW layout: channel = (linear_idx / (H*W)) % C
__global__ void fused_bias_hardswish_relu_kernel(
    const float* __restrict__ in,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t n,
    int HW,
    int C
) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int c = (int)(idx / HW) % C;
        float v = __ldg(in + idx) + __ldg(bias + c);
        // relu(hardswish(v)): x<=0->0, 0<x<3->x*(x+3)/6, x>=3->x
        float result = (v <= 0.f) ? 0.f : ((v >= 3.f) ? v : v * (v + 3.f) * (1.f / 6.f));
        out[idx] = result;
    }
}

torch::Tensor fused_bias_hardswish_relu(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "fused_bias_hardswish_relu: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "fused_bias_hardswish_relu: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "fused_bias_hardswish_relu: input must be contiguous");
    TORCH_CHECK(bias.is_cuda(), "fused_bias_hardswish_relu: bias must be a CUDA tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat, "fused_bias_hardswish_relu: bias must be float32");
    TORCH_CHECK(bias.is_contiguous(), "fused_bias_hardswish_relu: bias must be contiguous");
    TORCH_CHECK(x.dim() == 4, "fused_bias_hardswish_relu: input must be 4D (NCHW)");
    TORCH_CHECK(bias.numel() == x.size(1),
        "fused_bias_hardswish_relu: bias length must equal x.size(1)");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    int C  = (int)x.size(1);
    int HW = (int)(x.size(2) * x.size(3));

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    // cap blocks to avoid launch failures on tiny tensors
    if (blocks == 0) blocks = 1;

    fused_bias_hardswish_relu_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        n, HW, C
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, applies HardSwish, and then ReLU.
        """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = torch.nn.functional.conv2d(
            x,
            self.conv.weight,
            None,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
                and self.conv.bias is not None and self.conv.bias.is_cuda):
            x = _stark_get_extension().fused_bias_hardswish_relu(
                x, self.conv.bias.contiguous()
            )
            _fused = True
        else:
            if self.conv.bias is not None:
                x = x + self.conv.bias.view(1, -1, 1, 1)
            x = torch.nn.functional.hardswish(x)
            _fused = False
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not _fused:
            x = torch.relu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
