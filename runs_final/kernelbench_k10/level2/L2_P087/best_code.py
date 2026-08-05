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
    return f'stark_cuda_l2_p87_{digest}'

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

at::Tensor conv2d_subtract_subtract_mish_cuda(at::Tensor x, double subtract1, double subtract2);

at::Tensor conv2d_subtract_subtract_mish(at::Tensor x, double subtract1, double subtract2) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    return conv2d_subtract_subtract_mish_cuda(x, subtract1, subtract2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_subtract_subtract_mish", &conv2d_subtract_subtract_mish,
          "Fused subtract-subtract-mish activation (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_sub_sub_mish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float bias,
    int64_t n_vec,
    int64_t n_tail,
    int64_t total
) {
    // float4 vectorized path
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += gridDim.x * blockDim.x) {
        float4 v = in4[i];
        float x0 = v.x - bias;
        float x1 = v.y - bias;
        float x2 = v.z - bias;
        float x3 = v.w - bias;
        // Mish: x * tanh(softplus(x)), softplus(x) = log1p(exp(x)) clamped for stability
        float sp0 = x0 > 20.0f ? x0 : log1pf(expf(x0));
        float sp1 = x1 > 20.0f ? x1 : log1pf(expf(x1));
        float sp2 = x2 > 20.0f ? x2 : log1pf(expf(x2));
        float sp3 = x3 > 20.0f ? x3 : log1pf(expf(x3));
        v.x = x0 * tanhf(sp0);
        v.y = x1 * tanhf(sp1);
        v.z = x2 * tanhf(sp2);
        v.w = x3 * tanhf(sp3);
        out4[i] = v;
    }

    // scalar tail
    int64_t tail_start = n_vec * 4;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_tail; i += gridDim.x * blockDim.x) {
        int64_t idx = tail_start + i;
        float x0 = input[idx] - bias;
        float sp = x0 > 20.0f ? x0 : log1pf(expf(x0));
        output[idx] = x0 * tanhf(sp);
    }
}

at::Tensor conv2d_subtract_subtract_mish_cuda(at::Tensor x, double subtract1, double subtract2) {
    // Fast path: contiguous float32 CUDA tensor
    if (x.is_contiguous() && x.scalar_type() == at::kFloat) {
        auto out = at::empty_like(x);
        int64_t total = x.numel();
        float bias = (float)(subtract1 + subtract2);
        int64_t n_vec = total / 4;
        int64_t n_tail = total % 4;

        const int threads = 256;
        int blocks = (int)((n_vec + threads - 1) / threads);
        // Ensure we cover tail elements too
        if (n_tail > 0 && blocks == 0) blocks = 1;
        // Cap blocks
        if (blocks > 65535) blocks = 65535;

        fused_sub_sub_mish_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            bias,
            n_vec,
            n_tail,
            total
        );
        return out;
    }
    // Fallback: ATen ops for non-contiguous or non-float32
    auto xc = x.contiguous();
    xc = xc - (float)(subtract1 + subtract2);
    return at::mish(xc);
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, subtracts two values, applies Mish activation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        x = self.conv(x)
        x = _stark_get_extension().conv2d_subtract_subtract_mish(x, float(self.subtract_value_1), float(self.subtract_value_2))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        # subtract_value_1 fused into custom CUDA epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # subtract_value_2 fused into custom CUDA epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # Mish fused into custom CUDA epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
