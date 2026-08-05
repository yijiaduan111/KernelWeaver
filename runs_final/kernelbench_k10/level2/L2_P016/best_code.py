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
    return f'stark_cuda_l2_p16_{digest}'

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

void fused_epilogue(torch::Tensor x, float add_value, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_epilogue", &fused_epilogue, "Fused Mish+Add+Hardtanh+Scale epilogue (in-place, CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused epilogue: Mish -> add -> Hardtanh([-1,1]) -> scale, in-place over float32 tensor.
// Uses float4 vectorized loads/stores for throughput.
__global__ void fused_mish_add_hardtanh_scale_kernel(
    float* __restrict__ data,
    int64_t n4,        // number of float4 groups
    int64_t remainder, // leftover elements
    float add_value,
    float scale
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n4) {
        float4 v = reinterpret_cast<float4*>(data)[idx];

        #pragma unroll
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            float x = vals[i];
            // Stable softplus: avoid overflow for large x
            float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
            // Mish
            x = x * tanhf(sp);
            // Add
            x = x + add_value;
            // Hardtanh
            x = fmaxf(-1.0f, fminf(1.0f, x));
            // Scale
            x = x * scale;
            vals[i] = x;
        }
        v.x = vals[0]; v.y = vals[1]; v.z = vals[2]; v.w = vals[3];
        reinterpret_cast<float4*>(data)[idx] = v;
    }

    // Handle remainder elements
    int64_t base = n4 * 4;
    if (idx < remainder) {
        float x = data[base + idx];
        float sp = x > 20.0f ? x : __logf(1.0f + __expf(x));
        x = x * tanhf(sp);
        x = x + add_value;
        x = fmaxf(-1.0f, fminf(1.0f, x));
        x = x * scale;
        data[base + idx] = x;
    }
}

void fused_epilogue(torch::Tensor x, float add_value, float scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    int64_t numel = x.numel();
    int64_t n4 = numel / 4;
    int64_t remainder = numel % 4;

    int64_t nthreads = 256;
    int64_t nblocks = (n4 + nthreads - 1) / nthreads;
    // Ensure at least 1 block if there are remainder elements
    if (nblocks == 0 && remainder > 0) nblocks = 1;

    if (nblocks > 0) {
        fused_mish_add_hardtanh_scale_kernel<<<nblocks, nthreads>>>(
            x.data_ptr<float>(),
            n4,
            remainder,
            add_value,
            scale
        );
    }
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed convolution, applies Mish activation, adds a value, 
        applies Hardtanh activation, and scales the output.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        x = self.conv_transpose(x)
        x = x.contiguous()
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _stark_get_extension().fused_epilogue(x, float(self.add_value), float(self.scale))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
