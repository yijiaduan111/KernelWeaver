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
    return f'stark_cuda_l2_p48_{digest}'

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

torch::Tensor fused_post_conv(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused scale+tanh+bias+sigmoid epilogue");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__launch_bounds__(256, 4)
__global__ void fused_post_conv_kernel_scalar(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ scale,
    const float* __restrict__ bias_vec,
    int C,
    int spatial,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / spatial) % C;
    float s = scale[c];
    float b = bias_vec[c];
    float v = x[idx];
    v = v * s;
    v = tanhf(v);
    v = v * b;
    v = __fdividef(1.0f, 1.0f + __expf(-v));
    out[idx] = v;
}

// Channel-major vec4 kernel: blockIdx.y = nc slice, blockIdx.x covers spatial float4 chunks.
// scale[c] and bias_vec[c] are loaded once per block, eliminating per-lane integer division.
__launch_bounds__(256, 4)
__global__ void fused_post_conv_kernel_vec4(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    const float* __restrict__ scale,
    const float* __restrict__ bias_vec,
    int C,
    int spatial_vec4  // spatial / 4
) {
    int nc = blockIdx.y;  // flattened (n, c) slice index
    int c = nc % C;
    // Load scale and bias once per block into registers
    float s = scale[c];
    float b = bias_vec[c];

    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= spatial_vec4) return;

    int global_vec_idx = nc * spatial_vec4 + vec_idx;
    float4 v4 = x[global_vec_idx];

    float4 r;
    r.x = __fdividef(1.0f, 1.0f + __expf(-(tanhf(v4.x * s) * b)));
    r.y = __fdividef(1.0f, 1.0f + __expf(-(tanhf(v4.y * s) * b)));
    r.z = __fdividef(1.0f, 1.0f + __expf(-(tanhf(v4.z * s) * b)));
    r.w = __fdividef(1.0f, 1.0f + __expf(-(tanhf(v4.w * s) * b)));

    out[global_vec_idx] = r;
}

torch::Tensor fused_post_conv(
    torch::Tensor x,
    torch::Tensor scaling_factor,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    int N = x.size(0);
    int C = x.size(1);
    int total = (int)x.numel();
    int spatial = total / (N * C);
    auto scale_c = scaling_factor.contiguous().view({-1});
    auto bias_c = bias.contiguous().view({-1});
    auto out = torch::empty_like(x);
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    // vec4 path requires: spatial divisible by 4 (ensures per-slice alignment),
    // and both pointers 16-byte aligned.
    bool use_vec4 = (spatial % 4 == 0) &&
                    (reinterpret_cast<uintptr_t>(x_ptr) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(out_ptr) % 16 == 0);
    if (use_vec4) {
        int spatial_vec4 = spatial / 4;
        const int threads = 256;
        int blocks_x = (spatial_vec4 + threads - 1) / threads;
        int blocks_y = N * C;
        dim3 grid(blocks_x, blocks_y);
        fused_post_conv_kernel_vec4<<<grid, threads>>>(
            reinterpret_cast<const float4*>(x_ptr),
            reinterpret_cast<float4*>(out_ptr),
            scale_c.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            C, spatial_vec4
        );
    } else {
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        fused_post_conv_kernel_scalar<<<blocks, threads>>>(
            x_ptr,
            out_ptr,
            scale_c.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            C, spatial, total
        );
    }
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
        """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda:
            x = _stark_get_extension().fused_post_conv(x.contiguous(), self.scaling_factor.contiguous(), self.bias.contiguous())
            _fused = True
        else:
            x = x * self.scaling_factor
            _fused = False
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not _fused:
                    x = torch.tanh(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not _fused:
                    x = x * self.bias
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if not _fused:
                    x = torch.sigmoid(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
