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
    return f'stark_cuda_l2_p82_{digest}'

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

torch::Tensor fused_tanh_scale_bias_maxpool(
    torch::Tensor x,
    double scaling_factor,
    torch::Tensor bias,
    int64_t pool_kernel_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh_scale_bias_maxpool", &fused_tanh_scale_bias_maxpool,
          "Fused tanh/scale/bias/maxpool (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_tanh_scale_bias_maxpool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    float scaling_factor,
    int pool_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c  = (idx / (W_out * H_out)) % C;
    int n  = idx / (W_out * H_out * C);

    float max_val = -3.402823466e+38f;
    int ih_base = oh * pool_size;
    int iw_base = ow * pool_size;
    float b = __ldg(&bias[c]);

    for (int kh = 0; kh < pool_size; kh++) {
        int ih = ih_base + kh;
        if (ih >= H) continue;
        for (int kw = 0; kw < pool_size; kw++) {
            int iw = iw_base + kw;
            if (iw >= W) continue;
            float v = input[((n * C + c) * H + ih) * W + iw];
            float val = tanhf(v) * scaling_factor + b;
            max_val = fmaxf(max_val, val);
        }
    }

    output[((n * C + c) * H_out + oh) * W_out + ow] = max_val;
}

torch::Tensor fused_tanh_scale_bias_maxpool(
    torch::Tensor x,
    double scaling_factor,
    torch::Tensor bias,
    int64_t pool_kernel_size
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int pool = (int)pool_kernel_size;
    const int H_out = (H - pool) / pool + 1;
    const int W_out = (W - pool) / pool + 1;

    auto bias_c = bias.contiguous();
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    const int total = N * C * H_out * W_out;
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;

    fused_tanh_scale_bias_maxpool_kernel<<<gridSize, blockSize, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        bias_c.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, H_out, W_out,
        (float)scaling_factor,
        pool
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
        """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        pool_k = self.max_pool.kernel_size if isinstance(self.max_pool.kernel_size, int) else self.max_pool.kernel_size[0]
        x = _stark_get_extension().fused_tanh_scale_bias_maxpool(x.contiguous(), float(self.scaling_factor), self.bias.contiguous(), pool_k)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
