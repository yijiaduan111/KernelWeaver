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
    return f'stark_cuda_l2_p36_{digest}'

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

torch::Tensor bias_add_float4(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bias_add_float4", &bias_add_float4, "Vectorized broadcast bias add (float4 fast path)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Vectorized bias-add kernel: each thread processes 4 floats.
// bias_numel elements broadcast over x_numel output elements.
__global__ void bias_add_float4_kernel(
    const float4* __restrict__ x_vec,
    const float*  __restrict__ bias,
    float4*       __restrict__ out_vec,
    int64_t vec_count,
    int64_t bias_numel
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= vec_count) return;
    float4 xv = __ldg(&x_vec[idx]);
    int64_t base = idx * 4;
    float b0 = __ldg(&bias[(base + 0) % bias_numel]);
    float b1 = __ldg(&bias[(base + 1) % bias_numel]);
    float b2 = __ldg(&bias[(base + 2) % bias_numel]);
    float b3 = __ldg(&bias[(base + 3) % bias_numel]);
    xv.x += b0; xv.y += b1; xv.z += b2; xv.w += b3;
    out_vec[idx] = xv;
}

// Scalar tail/fallback kernel.
__global__ void bias_add_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float*       __restrict__ out,
    int64_t numel,
    int64_t bias_numel,
    int64_t offset
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[offset + idx] = __ldg(&x[offset + idx]) + __ldg(&bias[(offset + idx) % bias_numel]);
}

torch::Tensor bias_add_float4(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "bias_add_float4: tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat && bias.scalar_type() == torch::kFloat,
                "bias_add_float4: only float32 supported");

    auto x_contig    = x.contiguous();
    auto bias_contig = bias.contiguous();
    auto out         = torch::empty_like(x_contig);

    int64_t numel      = x_contig.numel();
    int64_t bias_numel = bias_contig.numel();

    const float* x_ptr    = x_contig.data_ptr<float>();
    const float* bias_ptr = bias_contig.data_ptr<float>();
    float*       out_ptr  = out.data_ptr<float>();

    // Check 16-byte alignment and numel multiple of 4 for vectorized path.
    bool aligned = ((reinterpret_cast<uintptr_t>(x_ptr)   & 15) == 0) &&
                   ((reinterpret_cast<uintptr_t>(out_ptr)  & 15) == 0) &&
                   (numel % 4 == 0);

    constexpr int BLOCK = 256;

    if (aligned) {
        int64_t vec_count = numel / 4;
        int64_t grid = (vec_count + BLOCK - 1) / BLOCK;
        bias_add_float4_kernel<<<grid, BLOCK>>>(
            reinterpret_cast<const float4*>(x_ptr),
            bias_ptr,
            reinterpret_cast<float4*>(out_ptr),
            vec_count,
            bias_numel
        );
    } else {
        // Full scalar fallback.
        int64_t grid = (numel + BLOCK - 1) / BLOCK;
        bias_add_scalar_kernel<<<grid, BLOCK>>>(x_ptr, bias_ptr, out_ptr, numel, bias_numel, 0);
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        if x.is_cuda:
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.to(memory_format=torch.channels_last)
            x = self.conv_transpose(x)
        else:
            x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = torch.amin(x, dim=1, keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = torch.sum(x, dim=2, keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = torch.nn.functional.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = _stark_get_extension().bias_add_float4(x, self.bias)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
