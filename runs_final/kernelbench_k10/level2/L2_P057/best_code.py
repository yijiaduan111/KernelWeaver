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
    return f'stark_cuda_l2_p57_{digest}'

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

torch::Tensor fused_relu_hardswish(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu_hardswish", &fused_relu_hardswish, "Fused ReLU + HardSwish (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_relu_hardswish_vec(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int nvec
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvec) return;
    float4 v = reinterpret_cast<const float4*>(inp)[idx];
    float4 o;
    float r;
    // relu(x) * clamp((relu(x)+3)/6, 0, 1)
    // since relu(x)>=0, lower clamp never fires; (r+3)/6 = fma(r, 1/6, 0.5)
    r = fmaxf(v.x, 0.0f); o.x = r * fminf(__fmaf_rn(r, 0.16666667f, 0.5f), 1.0f);
    r = fmaxf(v.y, 0.0f); o.y = r * fminf(__fmaf_rn(r, 0.16666667f, 0.5f), 1.0f);
    r = fmaxf(v.z, 0.0f); o.z = r * fminf(__fmaf_rn(r, 0.16666667f, 0.5f), 1.0f);
    r = fmaxf(v.w, 0.0f); o.w = r * fminf(__fmaf_rn(r, 0.16666667f, 0.5f), 1.0f);
    reinterpret_cast<float4*>(out)[idx] = o;
}

__global__ void fused_relu_hardswish_tail(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int start,
    int n
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float r = fmaxf(inp[idx], 0.0f);
    out[idx] = r * fminf(__fmaf_rn(r, 0.16666667f, 0.5f), 1.0f);
}

torch::Tensor fused_relu_hardswish(torch::Tensor input) {
    if (!input.is_cuda() || input.scalar_type() != torch::kFloat32) {
        auto x = torch::relu(input);
        return x * torch::clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f);
    }
    auto x = input.contiguous();
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    int64_t nvec = n / 4;
    int64_t rem  = n % 4;
    const int BS = 256;
    if (nvec > 0) {
        int grid = (int)((nvec + BS - 1) / BS);
        fused_relu_hardswish_vec<<<grid, BS>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), (int)nvec);
    }
    if (rem > 0) {
        fused_relu_hardswish_tail<<<1, (int)rem>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), (int)(nvec * 4), (int)n);
    }
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a convolution, applies ReLU, and applies HardSwish activation.
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
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_relu_hardswish(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
