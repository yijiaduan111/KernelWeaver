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
    return f'stark_cuda_l2_p2_{digest}'

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

torch::Tensor fused_post_ops_cuda(torch::Tensor x, torch::Tensor bias, double scaling_factor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_ops", &fused_post_ops_cuda, "Fused bias+clamp+scale+clamp+divide (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// In-place scalar kernel
__global__ void fused_post_ops_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int64_t numel,
    int64_t HW,
    int64_t C,
    float scale,
    float inv_scale
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int64_t c = (idx / HW) % C;
    float v = x[idx] + __ldg(&bias[c]);
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    v = v * scale;
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    x[idx] = v * inv_scale;
}

// In-place float4 vectorized kernel - 4 floats per thread via 128-bit loads/stores
__global__ void fused_post_ops_kernel_vec4(
    float4* __restrict__ x4,
    const float* __restrict__ bias,
    int64_t npacks,
    int64_t HW,
    int64_t C,
    float scale,
    float inv_scale
) {
    int64_t pid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= npacks) return;

    int64_t base = pid * 4;
    float4 v = x4[pid];

    {
        int64_t c = ((base + 0) / HW) % C;
        float val = v.x + __ldg(&bias[c]);
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scale;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        v.x = val * inv_scale;
    }
    {
        int64_t c = ((base + 1) / HW) % C;
        float val = v.y + __ldg(&bias[c]);
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scale;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        v.y = val * inv_scale;
    }
    {
        int64_t c = ((base + 2) / HW) % C;
        float val = v.z + __ldg(&bias[c]);
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scale;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        v.z = val * inv_scale;
    }
    {
        int64_t c = ((base + 3) / HW) % C;
        float val = v.w + __ldg(&bias[c]);
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scale;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        v.w = val * inv_scale;
    }

    x4[pid] = v;
}

torch::Tensor fused_post_ops_cuda(torch::Tensor x, torch::Tensor bias, double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t HW = H * W;
    int64_t numel = N * C * HW;

    float scale = (float)scaling_factor;
    float inv_scale = 1.0f / scale;

    float* xptr = x.data_ptr<float>();
    const float* biasptr = bias.data_ptr<float>();

    const int block = 256;

    bool use_vec4 = (reinterpret_cast<uintptr_t>(xptr) % 16 == 0) &&
                    (numel % 4 == 0);

    if (use_vec4) {
        int64_t npacks = numel / 4;
        int64_t grid = (npacks + block - 1) / block;
        fused_post_ops_kernel_vec4<<<grid, block>>>(
            reinterpret_cast<float4*>(xptr),
            biasptr,
            npacks,
            HW,
            C,
            scale,
            inv_scale
        );
    } else {
        int64_t grid = (numel + block - 1) / block;
        fused_post_ops_kernel<<<grid, block>>>(
            xptr,
            biasptr,
            numel,
            HW,
            C,
            scale,
            inv_scale
        );
    }

    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_post_ops(x.contiguous(), self.bias.reshape(-1).contiguous(), float(self.scaling_factor))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
