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
    return f'stark_cuda_l2_p26_{digest}'

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

torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_hardswish", &fused_add_hardswish_cuda, "Fused add and hardswish with shape dispatch (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Vectorized kernel: processes 4 floats per thread using float4.
// Requires x and add_input to have the same shape and be contiguous float32.
__global__ void __launch_bounds__(256) fused_add_hardswish_vec4_kernel(
    const float4* __restrict__ x,
    const float4* __restrict__ add_input,
    float4* __restrict__ out,
    int64_t num_vec4
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vec4) return;

    float4 xv = x[idx];
    float4 av = add_input[idx];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float v;
        if (i == 0) v = xv.x + av.x;
        else if (i == 1) v = xv.y + av.y;
        else if (i == 2) v = xv.z + av.z;
        else v = xv.w + av.w;

        float hs = v * fminf(fmaxf(v + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
        float res = v * hs;

        if (i == 0) xv.x = res;
        else if (i == 1) xv.y = res;
        else if (i == 2) xv.z = res;
        else xv.w = res;
    }
    out[idx] = xv;
}

// Scalar tail kernel for remainder elements in the same-shape path.
__global__ void fused_add_hardswish_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add_input,
    float* __restrict__ out,
    int64_t offset,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= total) return;
    float v = x[idx] + add_input[idx];
    float hs = v * fminf(fmaxf(v + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
    out[idx] = v * hs;
}

// Channel-broadcast kernel: add_input has exactly C values (one per output channel).
// x layout: (N, C, D, H, W) contiguous; spatial_size = D*H*W.
// For flat index i: channel = (i / spatial_size) % C.
__global__ void __launch_bounds__(256) fused_add_hardswish_chan_bcast_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add_channel,
    float* __restrict__ out,
    int64_t total,
    int64_t spatial_size,
    int64_t C
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t c = (idx / spatial_size) % C;
    float v = x[idx] + __ldg(add_channel + c);
    float hs = v * fminf(fmaxf(v + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
    out[idx] = v * hs;
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input) {
    TORCH_CHECK(x.is_cuda() && add_input.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.is_contiguous() && add_input.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && add_input.dtype() == torch::kFloat32, "Tensors must be float32");

    auto out = torch::empty_like(x);
    int64_t total = x.numel();
    const int threads = 256;

    if (x.sizes() == add_input.sizes()) {
        // Exact same-shape path: float4 vectorized
        int64_t num_vec4 = total / 4;
        int64_t remainder = total % 4;
        if (num_vec4 > 0) {
            int blocks = (int)((num_vec4 + threads - 1) / threads);
            fused_add_hardswish_vec4_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(x.data_ptr<float>()),
                reinterpret_cast<const float4*>(add_input.data_ptr<float>()),
                reinterpret_cast<float4*>(out.data_ptr<float>()),
                num_vec4
            );
        }
        if (remainder > 0) {
            int64_t offset = num_vec4 * 4;
            int blocks_r = (int)((remainder + threads - 1) / threads);
            fused_add_hardswish_scalar_kernel<<<blocks_r, threads>>>(
                x.data_ptr<float>(), add_input.data_ptr<float>(), out.data_ptr<float>(), offset, total
            );
        }
    } else {
        // Channel-broadcast path: add_input.numel() == C (broadcast over N and spatial dims)
        TORCH_CHECK(x.dim() >= 2 && add_input.numel() == x.size(1),
                    "add_input must match x shape or have numel == x.size(1) for channel broadcast");
        int64_t C = x.size(1);
        int64_t spatial_size = total / (x.size(0) * C);
        int blocks = (int)((total + threads - 1) / threads);
        fused_add_hardswish_chan_bcast_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            add_input.data_ptr<float>(),
            out.data_ptr<float>(),
            total,
            spatial_size,
            C
        );
    }
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x, add_input):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
                    add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D, H, W).
                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W) after HardSwish activation.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        _is_exact_shape = (
        x.is_cuda and add_input.is_cuda
        and x.is_contiguous() and add_input.is_contiguous()
        and x.dtype == torch.float32 and add_input.dtype == torch.float32
        and x.shape == add_input.shape
        )
        _is_chan_bcast = (
        not _is_exact_shape
        and x.is_cuda and add_input.is_cuda
        and x.is_contiguous() and add_input.is_contiguous()
        and x.dtype == torch.float32 and add_input.dtype == torch.float32
        and x.dim() >= 2 and add_input.numel() == x.size(1)
        )
        _use_cuda_fused = _is_exact_shape or _is_chan_bcast
        if _use_cuda_fused:
            x = _stark_get_extension().fused_add_hardswish(x, add_input)
        else:
            x = x + add_input
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not _use_cuda_fused:
                    x = x * torch.nn.functional.hardswish(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
