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
    return f'stark_cuda_l2_p93_{digest}'

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

void fused_post_conv_cuda(torch::Tensor x, float add_value, float multiply_value);

void fused_post_conv(torch::Tensor x, float add_value, float multiply_value) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be a dense (non-overlapping) tensor");
    fused_post_conv_cuda(x, add_value, multiply_value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv epilogue (add, min(x,0), gelu, multiply)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float apply_epilogue(float v, float add_value, float multiply_value) {
    v = v + add_value;
    v = v < 0.0f ? v : 0.0f;
    v = fast_gelu(v);
    v = v * multiply_value;
    return v;
}

// Vectorized kernel: processes 4 floats per thread via float4
__global__ void __launch_bounds__(256, 4)
fused_post_conv_vec4_kernel(float4* __restrict__ xv, int64_t numel4, float add_value, float multiply_value) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= numel4) return;
    float4 v = __ldg(xv + idx);
    v.x = apply_epilogue(v.x, add_value, multiply_value);
    v.y = apply_epilogue(v.y, add_value, multiply_value);
    v.z = apply_epilogue(v.z, add_value, multiply_value);
    v.w = apply_epilogue(v.w, add_value, multiply_value);
    xv[idx] = v;
}

// Scalar tail kernel for remaining elements
__global__ void __launch_bounds__(256, 4)
fused_post_conv_tail_kernel(float* __restrict__ x, int64_t start, int64_t numel, float add_value, float multiply_value) {
    int64_t idx = start + (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= numel) return;
    float v = __ldg(x + idx);
    v = apply_epilogue(v, add_value, multiply_value);
    x[idx] = v;
}

void fused_post_conv_cuda(torch::Tensor x, float add_value, float multiply_value) {
    int64_t numel = x.numel();
    float* ptr = x.data_ptr<float>();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    bool aligned16 = (addr % 16 == 0);

    if (aligned16 && numel >= 4) {
        int64_t numel4 = numel / 4;
        int64_t tail_start = numel4 * 4;
        int64_t tail_count = numel - tail_start;

        // Launch vectorized kernel
        int64_t blocks4 = (numel4 + 255) / 256;
        fused_post_conv_vec4_kernel<<<blocks4, 256>>>(
            reinterpret_cast<float4*>(ptr),
            numel4,
            add_value,
            multiply_value
        );

        // Handle tail elements (< 4)
        if (tail_count > 0) {
            fused_post_conv_tail_kernel<<<1, 256>>>(
                ptr, tail_start, numel, add_value, multiply_value
            );
        }
    } else {
        // Scalar fallback
        int64_t blocks = (numel + 255) / 256;
        fused_post_conv_tail_kernel<<<blocks, 256>>>(
            ptr, 0, numel, add_value, multiply_value
        );
    }
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self._prefer_channels_last = True
        try:
            w = self.conv_transpose.weight.detach().contiguous(memory_format=torch.channels_last)
            self.conv_transpose.weight = nn.Parameter(w, requires_grad=self.conv_transpose.weight.requires_grad)
            if self.conv_transpose.bias is not None:
                b = self.conv_transpose.bias.detach().contiguous()
                self.conv_transpose.bias = nn.Parameter(b, requires_grad=self.conv_transpose.bias.requires_grad)
            self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)
        except Exception:
            self._prefer_channels_last = False
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        if self._prefer_channels_last and x.is_cuda and x.dim() == 4 and x.dtype == torch.float32:
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
            x = self.conv_transpose(x)
        else:
            x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _stark_get_extension().fused_post_conv(x, self.add_value, self.multiply_value)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # fused into fused_post_conv kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # fused into fused_post_conv kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # fused into fused_post_conv kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
