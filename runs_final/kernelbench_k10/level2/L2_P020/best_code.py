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
    return f'stark_cuda_l2_p20_{digest}'

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

torch::Tensor fused_pointwise_epilogue(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_pointwise_epilogue", &fused_pointwise_epilogue, "Fused pointwise epilogue");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_pointwise_kernel_vectorized(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t numel_vec,
    int64_t C,
    int64_t spatial_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel_vec) {
        float4 val4 = reinterpret_cast<const float4*>(x)[idx];
        int64_t base_idx = idx * 4;
        int64_t c0 = (base_idx / spatial_size) % C;
        int64_t c1 = ((base_idx + 1) / spatial_size) % C;
        int64_t c2 = ((base_idx + 2) / spatial_size) % C;
        int64_t c3 = ((base_idx + 3) / spatial_size) % C;

        float b0 = bias[c0];
        float b1 = bias[c1];
        float b2 = bias[c2];
        float b3 = bias[c3];

        float v0 = val4.x;
        float v1 = val4.y;
        float v2 = val4.z;
        float v3 = val4.w;

        float r0 = ((v0 + b0) + v0) * v0 + v0;
        float r1 = ((v1 + b1) + v1) * v1 + v1;
        float r2 = ((v2 + b2) + v2) * v2 + v2;
        float r3 = ((v3 + b3) + v3) * v3 + v3;

        reinterpret_cast<float4*>(out)[idx] = make_float4(r0, r1, r2, r3);
    }
}

__global__ void fused_pointwise_kernel_scalar(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t numel,
    int64_t C,
    int64_t spatial_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int64_t c_idx = (idx / spatial_size) % C;
        float val = x[idx];
        float b = bias[c_idx];
        float tmp = val + b;
        tmp = tmp + val;
        tmp = tmp * val;
        out[idx] = tmp + val;
    }
}

torch::Tensor fused_pointwise_epilogue(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 5, "x must be 5D (N, C, D, H, W)");

    int64_t N = sizes[0];
    int64_t C = sizes[1];
    int64_t D = sizes[2];
    int64_t H = sizes[3];
    int64_t W = sizes[4];
    int64_t spatial_size = D * H * W;
    int64_t numel = N * C * spatial_size;

    TORCH_CHECK(bias.size(0) == C, "bias must have size C in dimension 0");

    auto out = torch::empty_like(x);

    const int threads = 256;

    uintptr_t x_addr = reinterpret_cast<uintptr_t>(x.data_ptr<float>());
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out.data_ptr<float>());
    bool aligned = (x_addr % 16 == 0) && (out_addr % 16 == 0);
    bool vec_eligible = aligned && (numel % 4 == 0);

    if (vec_eligible) {
        int64_t numel_vec = numel / 4;
        const int blocks = (numel_vec + threads - 1) / threads;
        fused_pointwise_kernel_vectorized<<<blocks, threads>>>(
            x.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            numel_vec,
            C,
            spatial_size
        );
    } else {
        const int blocks = (numel + threads - 1) / threads;
        fused_pointwise_kernel_scalar<<<blocks, threads>>>(
            x.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            numel,
            C,
            spatial_size
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, followed by a sum, 
        a residual add, a multiplication, and another residual add.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        original_x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().fused_pointwise_epilogue(x, self.bias)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
