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
    return f'stark_cuda_l2_p17_{digest}'

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

torch::Tensor fused_instancenorm_divide(torch::Tensor input, double divide_by);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_instancenorm_divide", &fused_instancenorm_divide, "Fused InstanceNorm and divide");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
__global__ void fused_instancenorm_divide_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t num_groups,
    int64_t spatial,
    float divide_by,
    float eps) {
    const int group = blockIdx.x;
    if (group >= num_groups) {
        return;
    }

    const float* in_ptr = input + group * spatial;
    float* out_ptr = output + group * spatial;

    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sumsq_shared = shared + blockDim.x;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < spatial; i += blockDim.x) {
        float v = in_ptr[i];
        local_sum += v;
        local_sumsq += v * v;
    }

    sum_shared[threadIdx.x] = local_sum;
    sumsq_shared[threadIdx.x] = local_sumsq;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sum_shared[threadIdx.x] += sum_shared[threadIdx.x + offset];
            sumsq_shared[threadIdx.x] += sumsq_shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float mean = sum_shared[0] / static_cast<float>(spatial);
        const float var = sumsq_shared[0] / static_cast<float>(spatial) - mean * mean;
        const float inv_std = rsqrtf(var + eps);
        sum_shared[0] = mean;
        sumsq_shared[0] = inv_std / divide_by;
    }
    __syncthreads();

    const float mean = sum_shared[0];
    const float scale = sumsq_shared[0];
    for (int64_t i = threadIdx.x; i < spatial; i += blockDim.x) {
        out_ptr[i] = (in_ptr[i] - mean) * scale;
    }
}
} // namespace

torch::Tensor fused_instancenorm_divide(torch::Tensor input, double divide_by) {
    TORCH_CHECK(input.is_cuda(), "fused_instancenorm_divide expects a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "fused_instancenorm_divide expects float32 input");
    TORCH_CHECK(input.dim() == 4, "fused_instancenorm_divide expects a 4D NCHW tensor");

    c10::cuda::CUDAGuard device_guard(input.device());
    input = input.contiguous();

    const auto n = input.size(0);
    const auto c = input.size(1);
    const auto h = input.size(2);
    const auto w = input.size(3);
    const int64_t num_groups = n * c;
    const int64_t spatial = h * w;

    auto output = torch::empty_like(input);
    const int threads = (spatial >= 8192) ? 512 : 256;
    const dim3 blocks(static_cast<unsigned int>(num_groups));
    const size_t shared_mem = static_cast<size_t>(threads) * 2u * sizeof(float);
    const float eps = 1e-5f;

    fused_instancenorm_divide_kernel<<<blocks, threads, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_groups,
        spatial,
        static_cast<float>(divide_by),
        eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
        """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_instancenorm_divide(x, float(self.divide_by))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # Fused into the CUDA extension call above.
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
