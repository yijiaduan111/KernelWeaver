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
    return f'stark_cuda_l1_p20_{digest}'

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

torch::Tensor leaky_relu_cuda(torch::Tensor x, double negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_cuda", &leaky_relu_cuda, "LeakyReLU CUDA launch-tuned float4");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float leaky_relu_op(float v, float slope) {
    return v >= 0.f ? v : v * slope;
}

__launch_bounds__(512, 2)
__global__ void leaky_relu_vec8_kernel(
    const float4* __restrict__ input,
    float4*       __restrict__ output,
    int64_t vec2_n,
    float negative_slope
) {
    int64_t idx    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    for (int64_t i = idx; i < vec2_n; i += stride) {
        // Process first float4, then second float4 sequentially to shorten live ranges
        float4 a = __ldg(&input[2*i]);
        a.x = leaky_relu_op(a.x, negative_slope);
        a.y = leaky_relu_op(a.y, negative_slope);
        a.z = leaky_relu_op(a.z, negative_slope);
        a.w = leaky_relu_op(a.w, negative_slope);
        output[2*i] = a;

        float4 b = __ldg(&input[2*i+1]);
        b.x = leaky_relu_op(b.x, negative_slope);
        b.y = leaky_relu_op(b.y, negative_slope);
        b.z = leaky_relu_op(b.z, negative_slope);
        b.w = leaky_relu_op(b.w, negative_slope);
        output[2*i+1] = b;
    }
}

__launch_bounds__(512, 2)
__global__ void leaky_relu_scalar_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int64_t n,
    float negative_slope
) {
    int64_t idx    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float v = input[i];
        output[i] = leaky_relu_op(v, negative_slope);
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "leaky_relu_cuda: input must be on CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "leaky_relu_cuda: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "leaky_relu_cuda: input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) return output;

    float slope = static_cast<float>(negative_slope);
    const float* in_ptr  = x.data_ptr<float>();
    float*       out_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    bool aligned = (reinterpret_cast<uintptr_t>(in_ptr)  % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(out_ptr) % 16 == 0);

    const int threads = 512;

    if (aligned && numel >= 8) {
        int64_t vec2_n   = numel / 8;
        int64_t tail_start = vec2_n * 8;
        int64_t tail_n   = numel - tail_start;

        int64_t blocks = (vec2_n + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;

        leaky_relu_vec8_kernel<<<(int)blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in_ptr),
            reinterpret_cast<float4*>(out_ptr),
            vec2_n,
            slope
        );

        if (tail_n > 0) {
            int64_t tail_blocks = (tail_n + threads - 1) / threads;
            if (tail_blocks > 65535) tail_blocks = 65535;
            leaky_relu_scalar_kernel<<<(int)tail_blocks, threads, 0, stream>>>(
                in_ptr  + tail_start,
                out_ptr + tail_start,
                tail_n,
                slope
            );
        }
    } else {
        int64_t blocks = (numel + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        leaky_relu_scalar_kernel<<<(int)blocks, threads, 0, stream>>>(
            in_ptr, out_ptr, numel, slope
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a LeakyReLU activation.
        """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the LeakyReLU module.

                Args:
                    negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
                """
        self.negative_slope = negative_slope
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies LeakyReLU activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of any shape.

                Returns:
                    torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return _stark_get_extension().leaky_relu_cuda(x, float(self.negative_slope))
        return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        # <<<END_IMPROVE>>>
