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
    return f'stark_cuda_l2_p52_{digest}'

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

torch::Tensor mish_cuda(torch::Tensor x);
torch::Tensor mish_bn_eval_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mish_cuda", &mish_cuda, "Mish activation (CUDA)");
    m.def("mish_bn_eval_cuda", &mish_bn_eval_cuda, "Fused Mish + BN eval (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float mish_f(float x) {
    // numerically safe: for large x, softplus(x) ~ x, tanh(x) ~ 1, mish(x) ~ x
    float sp = (x > 20.0f) ? x : log1pf(expf(x));
    return x * tanhf(sp);
}

__global__ void mish_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = mish_f(inp[idx]);
    }
}

__global__ void mish_bn_eval_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias_vec,
    float eps,
    int N, int C, int HW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (idx < total) {
        int c = (idx / HW) % C;
        float rm = __ldg(&running_mean[c]);
        float rv = __ldg(&running_var[c]);
        float w  = __ldg(&weight[c]);
        float b  = __ldg(&bias_vec[c]);
        float scale = w / sqrtf(rv + eps);
        float shift = b - scale * rm;
        float y = mish_f(inp[idx]);
        out[idx] = scale * y + shift;
    }
}

torch::Tensor mish_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mish_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mish_cuda: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "mish_cuda: input must be contiguous");
    auto out = torch::empty_like(x);
    int n = x.numel();
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n
    );
    return out;
}

torch::Tensor mish_bn_eval_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "mish_bn_eval_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mish_bn_eval_cuda: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "mish_bn_eval_cuda: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "mish_bn_eval_cuda: input must be 4D NCHW");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int HW = H * W;
    int total = N * C * HW;
    auto out = torch::empty_like(x);
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mish_bn_eval_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        running_mean.contiguous().data_ptr<float>(),
        running_var.contiguous().data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        bias.contiguous().data_ptr<float>(),
        (float)eps,
        N, C, HW
    );
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
        """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _use_custom = (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.dim() == 4)
        if _use_custom:
            ext = _stark_get_extension()
            if self.training:
                x = ext.mish_cuda(x)
                self._fused_eval_bn = False
            else:
                x = ext.mish_bn_eval_cuda(
                    x,
                    self.bn.running_mean,
                    self.bn.running_var,
                    self.bn.weight,
                    self.bn.bias,
                    self.bn.eps,
                )
                self._fused_eval_bn = True
        else:
            x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
            self._fused_eval_bn = False
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not getattr(self, '_fused_eval_bn', False):
            x = self.bn(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
