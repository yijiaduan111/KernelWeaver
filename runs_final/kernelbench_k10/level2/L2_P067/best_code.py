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
    return f'stark_cuda_l2_p67_{digest}'

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

torch::Tensor gelu_globalavgpool_cuda(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_globalavgpool_cuda", &gelu_globalavgpool_cuda, "Fused GELU + Global Average Pool with bias (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused bias-add + GELU (erf-based) + Global Average Pool kernel.
// Each block handles one (n, c) slice of the NCHW input tensor.
// Bias is added per-channel in registers before GELU, eliminating the standalone
// elementwise bias-add kernel that was the bandwidth bottleneck.
__global__ void gelu_gavgpool_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,   // shape (C,), may be nullptr
    float* __restrict__ output,
    int HW,
    int C
) {
    int nc = blockIdx.x;
    int c = nc % C;
    const float* base = input + nc * HW;

    // Load per-channel bias once into a register
    float b = (bias != nullptr) ? bias[c] : 0.0f;

    float partial = 0.0f;
    const float sqrt2inv = 0.7071067811865476f;

    // Check 16-byte alignment for float4 loads
    bool aligned = (reinterpret_cast<uintptr_t>(base) % 16 == 0);

    if (aligned && HW >= 4) {
        const float4* base4 = reinterpret_cast<const float4*>(base);
        int HW4 = HW >> 2;
        int tail_start = HW4 << 2;

        // Vectorized main loop with bias fold
        for (int i4 = threadIdx.x; i4 < HW4; i4 += blockDim.x) {
            float4 v4 = base4[i4];
            float x0 = v4.x + b;
            float x1 = v4.y + b;
            float x2 = v4.z + b;
            float x3 = v4.w + b;
            partial += 0.5f * x0 * (1.0f + erff(x0 * sqrt2inv));
            partial += 0.5f * x1 * (1.0f + erff(x1 * sqrt2inv));
            partial += 0.5f * x2 * (1.0f + erff(x2 * sqrt2inv));
            partial += 0.5f * x3 * (1.0f + erff(x3 * sqrt2inv));
        }

        // Scalar tail for remaining elements
        for (int i = tail_start + threadIdx.x; i < HW; i += blockDim.x) {
            float v = base[i] + b;
            partial += 0.5f * v * (1.0f + erff(v * sqrt2inv));
        }
    } else {
        // Fallback scalar loop
        for (int i = threadIdx.x; i < HW; i += blockDim.x) {
            float v = base[i] + b;
            partial += 0.5f * v * (1.0f + erff(v * sqrt2inv));
        }
    }

    // Warp-level reduction via shuffle
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(mask, partial, offset);
    }

    // Shared memory to collect warp sums
    __shared__ float smem[32]; // max 32 warps per block
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) smem[warp_id] = partial;
    __syncthreads();

    // First warp reduces warp sums
    int num_warps = (blockDim.x + 31) >> 5;
    float total = 0.0f;
    if (threadIdx.x < num_warps) {
        total = smem[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(mask, total, offset);
        }
    }

    if (threadIdx.x == 0) {
        output[nc] = total / static_cast<float>(HW);
    }
}

torch::Tensor gelu_globalavgpool_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;

    auto output = torch::empty({N, C}, x.options());

    // Determine bias pointer: nullptr if bias tensor is empty/undefined
    const float* bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias must be float32");
        TORCH_CHECK(bias.numel() == C, "Bias must have C elements");
        bias_ptr = bias.data_ptr<float>();
    }

    int block_size = 256;
    int grid_size = N * C;

    gelu_gavgpool_bias_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        HW,
        C
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a convolution, applies GELU, and then performs global average pooling.
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
        """
                Args:
                    x: Input tensor of shape (batch_size, in_channels, height, width)
                Returns:
                    Output tensor of shape (batch_size, out_channels)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = torch.nn.functional.conv2d(x, self.conv.weight, None, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        bias_tensor = self.conv.bias.contiguous() if self.conv.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        x = _stark_get_extension().gelu_globalavgpool_cuda(x.contiguous(), bias_tensor)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # Pooling is fused into the custom CUDA kernel above; no separate pooling needed.
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # Fused CUDA kernel already returns shape (batch_size, out_channels); no squeeze needed.
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
