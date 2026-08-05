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
    return f'stark_cuda_l1_p27_{digest}'

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

torch::Tensor selu_cuda(torch::Tensor x);

torch::Tensor selu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "selu_forward: input must be a CUDA tensor");
    return selu_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selu_forward", &selu_forward, "SELU forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static constexpr float SELU_ALPHA       = 1.6732632423543772f;
static constexpr float SELU_SCALE       = 1.0507009873554805f;
static constexpr float SELU_SCALE_ALPHA = SELU_SCALE * SELU_ALPHA;

__device__ __forceinline__ float selu_op(float x) {
    return x > 0.0f ? SELU_SCALE * x : SELU_SCALE_ALPHA * (__expf(x) - 1.0f);
}

__global__ void selu_kernel_ilp4(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int64_t n) {
    // Each thread handles 4 consecutive elements per grid-stride step
    int64_t base = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int64_t stride = (int64_t)gridDim.x * blockDim.x * 4;
    for (int64_t i = base; i < n; i += stride) {
        float x0, x1, x2, x3;
        x0 = (i + 0 < n) ? in[i + 0] : 0.0f;
        x1 = (i + 1 < n) ? in[i + 1] : 0.0f;
        x2 = (i + 2 < n) ? in[i + 2] : 0.0f;
        x3 = (i + 3 < n) ? in[i + 3] : 0.0f;
        float r0 = selu_op(x0);
        float r1 = selu_op(x1);
        float r2 = selu_op(x2);
        float r3 = selu_op(x3);
        if (i + 0 < n) out[i + 0] = r0;
        if (i + 1 < n) out[i + 1] = r1;
        if (i + 2 < n) out[i + 2] = r2;
        if (i + 3 < n) out[i + 3] = r3;
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "selu_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "selu_cuda: input must be contiguous");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    const int threads = 256;
    // Each thread covers 4 elements; cap grid at 65535
    int64_t groups = (n + 4 * threads - 1) / (4 * threads);
    int blocks = (int)std::min(groups, (int64_t)65535);

    selu_kernel_ilp4<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        n);
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a SELU activation.
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies SELU activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of any shape.

                Returns:
                    torch.Tensor: Output tensor with SELU applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return torch.selu(x)
        # <<<END_IMPROVE>>>
