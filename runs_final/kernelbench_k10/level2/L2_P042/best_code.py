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
    return f'stark_cuda_l2_p42_{digest}'

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

torch::Tensor fused_tail_cuda(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tail", &fused_tail_cuda, "Fused bias-add + logsumexp + scale (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel: bias-add + logsumexp over C channels + scale by 10.0
// Input x: [N, C, 1, 1] (contiguous), bias: [C] (or broadcastable to [C])
// Output: [N, 1]
// Block: one block per batch element, blockDim.x == C (assumed C <= 1024)
// Uses two-pass warp/block reduction for numerical stability.

template <int BLOCK>
__global__ void fused_bias_logsumexp_scale_kernel(
    const float* __restrict__ x,      // [N, C]
    const float* __restrict__ bias,   // [C]
    float* __restrict__ out,          // [N, 1]
    int C
) {
    int n = blockIdx.x;
    int c = threadIdx.x;

    // Load value + bias into register
    float val = (c < C) ? (x[n * C + c] + bias[c]) : -1e38f;

    // --- Pass 1: block-wide max reduction ---
    __shared__ float smem[BLOCK];
    smem[c] = val;
    __syncthreads();

    // Reduce max in shared memory
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (c < stride) {
            smem[c] = fmaxf(smem[c], smem[c + stride]);
        }
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    // --- Pass 2: block-wide sum(exp(v - max)) reduction ---
    smem[c] = (c < C) ? expf(val - max_val) : 0.0f;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (c < stride) {
            smem[c] += smem[c + stride];
        }
        __syncthreads();
    }

    if (c == 0) {
        float lse = logf(smem[0]) + max_val;
        out[n] = lse * 10.0f;
    }
}

torch::Tensor fused_tail_cuda(
    torch::Tensor x,     // [N, C, 1, 1]
    torch::Tensor bias   // [C, 1, 1] or [C]
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int N = x.size(0);
    int C = x.size(1);

    // Flatten x to [N, C] and bias to [C]
    auto x_flat = x.view({N, C});
    auto bias_flat = bias.reshape({C}).contiguous();

    auto out = torch::empty({N, 1}, x.options());

    // We specialize for C=128 (the benchmark case) with BLOCK=128
    // For other sizes, fall back to next power-of-two up to 512
    if (C == 128) {
        fused_bias_logsumexp_scale_kernel<128><<<N, 128>>>(
            x_flat.data_ptr<float>(),
            bias_flat.data_ptr<float>(),
            out.data_ptr<float>(),
            C
        );
    } else if (C <= 256) {
        fused_bias_logsumexp_scale_kernel<256><<<N, 256>>>(
            x_flat.data_ptr<float>(),
            bias_flat.data_ptr<float>(),
            out.data_ptr<float>(),
            C
        );
    } else {
        fused_bias_logsumexp_scale_kernel<512><<<N, 512>>>(
            x_flat.data_ptr<float>(),
            bias_flat.data_ptr<float>(),
            out.data_ptr<float>(),
            C
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
        """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Algebraic simplification: mean(ConvTranspose2d(x)) == (x_sum @ w_sum) / (H_out * W_out) + conv_bias
        # Valid for stride=1, padding=0, dilation=1, output_padding=0, groups=1 (all defaults here).
        ct = self.conv_transpose
        h_in, w_in = x.shape[2], x.shape[3]
        k_h, k_w = ct.kernel_size if isinstance(ct.kernel_size, tuple) else (ct.kernel_size, ct.kernel_size)
        stride_h, stride_w = ct.stride if isinstance(ct.stride, tuple) else (ct.stride, ct.stride)
        pad_h, pad_w = ct.padding if isinstance(ct.padding, tuple) else (ct.padding, ct.padding)
        dil_h, dil_w = ct.dilation if isinstance(ct.dilation, tuple) else (ct.dilation, ct.dilation)
        out_pad_h, out_pad_w = ct.output_padding if isinstance(ct.output_padding, tuple) else (ct.output_padding, ct.output_padding)
        h_out = (h_in - 1) * stride_h - 2 * pad_h + dil_h * (k_h - 1) + out_pad_h + 1
        w_out = (w_in - 1) * stride_w - 2 * pad_w + dil_w * (k_w - 1) + out_pad_w + 1
        x_sum = x.sum(dim=(2, 3))
        w_sum = ct.weight.sum(dim=(2, 3))
        x = x_sum.matmul(w_sum) / float(h_out * w_out)
        if ct.bias is not None:
            x = x + ct.bias
        x = x.unsqueeze(-1).unsqueeze(-1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().fused_tail(x, self.bias)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # logsumexp handled inside fused_tail
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # sum over singleton dims handled inside fused_tail
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # scale by 10.0 handled inside fused_tail
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
