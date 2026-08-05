import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    return f'stark_cuda_l1_p88_{digest}'

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

torch::Tensor gelu_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_forward", &gelu_forward, "Fused GELU forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
static constexpr float kBeta  = 0.044715f;

__device__ __forceinline__ float gelu_approx(float x) {
    float x2 = x * x;
    float inner = fmaf(kBeta * x2, x, x);  // x + 0.044715 * x^3 via FMA
    return 0.5f * x * (1.0f + __tanhf(kAlpha * inner));
}

__launch_bounds__(256, 4)
__global__ void gelu_kernel_float4(
    const float4* __restrict__ in,
    float4* __restrict__ out,
    int n4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = in[i];
        float4 r;
        r.x = gelu_approx(v.x);
        r.y = gelu_approx(v.y);
        r.z = gelu_approx(v.z);
        r.w = gelu_approx(v.w);
        out[i] = r;
    }
}

__launch_bounds__(256, 4)
__global__ void gelu_kernel_scalar(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = gelu_approx(in[i]);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_forward: input must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "gelu_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "gelu_forward: input must be contiguous");

    auto out = torch::empty_like(x);
    int64_t numel = x.numel();

    const float* in_ptr  = x.data_ptr<float>();
    float*       out_ptr = out.data_ptr<float>();

    bool aligned = (reinterpret_cast<uintptr_t>(in_ptr)  % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(out_ptr) % 16 == 0);

    const int threads = 256;

    if (aligned && (numel % 4 == 0)) {
        int n4 = numel / 4;
        int blocks = (n4 + threads - 1) / threads;
        blocks = min(blocks, 65535);
        gelu_kernel_float4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(in_ptr),
            reinterpret_cast<float4*>(out_ptr),
            n4);
    } else {
        int blocks = (numel + threads - 1) / threads;
        blocks = min(blocks, 65535);
        gelu_kernel_scalar<<<blocks, threads>>>(in_ptr, out_ptr, (int)numel);
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return _stark_get_extension().gelu_forward(x)
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        # <<<END_IMPROVE>>>
