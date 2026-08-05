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
    return f'stark_cuda_l1_p94_{digest}'

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

torch::Tensor fused_mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);

torch::Tensor fused_mse_loss(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == targets.scalar_type(), "predictions and targets must have the same dtype");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    return fused_mse_loss_cuda(predictions, targets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mse_loss", &fused_mse_loss, "Fused MSE loss (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Vectorized float32 kernel: processes 4 elements per thread via float4 loads
__global__ void mse_kernel_vec4(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ partial_sums,
    int64_t numel4,   // number of float4 groups
    int64_t numel     // total elements (for tail)
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* tgt4  = reinterpret_cast<const float4*>(targets);

    float thread_sum = 0.0f;

    // Vectorized loop over float4 groups
    for (int64_t i = idx; i < numel4; i += stride) {
        float4 p = __ldg(&pred4[i]);
        float4 t = __ldg(&tgt4[i]);
        float d0 = p.x - t.x; thread_sum += d0 * d0;
        float d1 = p.y - t.y; thread_sum += d1 * d1;
        float d2 = p.z - t.z; thread_sum += d2 * d2;
        float d3 = p.w - t.w; thread_sum += d3 * d3;
    }

    // Scalar tail (only first few threads handle remaining elements)
    int64_t tail_start = numel4 * 4;
    int64_t tail_len   = numel - tail_start;
    if (tid < tail_len) {
        float diff = predictions[tail_start + tid] - targets[tail_start + tid];
        thread_sum += diff * diff;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

template<typename scalar_t>
__global__ void mse_kernel_partial(
    const scalar_t* __restrict__ predictions,
    const scalar_t* __restrict__ targets,
    float* __restrict__ partial_sums,
    int64_t numel
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    float thread_sum = 0.0f;
    for (int64_t i = idx; i < numel; i += stride) {
        float diff = static_cast<float>(predictions[i]) - static_cast<float>(targets[i]);
        thread_sum += diff * diff;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_partials_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    int num_partials
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float thread_sum = 0.0f;
    for (int i = idx; i < num_partials; i += stride) {
        thread_sum += partial_sums[i];
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

torch::Tensor fused_mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int64_t numel = predictions.numel();

    const int threads = 256;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(predictions.device());
    auto output = torch::zeros({1}, options);

    // Check if we can use the float4 vectorized path:
    // float32, contiguous, 16-byte aligned pointers
    bool use_vec4 = (predictions.scalar_type() == at::ScalarType::Float) &&
                    (reinterpret_cast<uintptr_t>(predictions.data_ptr<float>()) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(targets.data_ptr<float>()) % 16 == 0);

    if (use_vec4) {
        int64_t numel4 = numel / 4;
        // Grid is sized over float4 groups; tail handled inside kernel by thread index
        int64_t grid_work = (numel4 > 0) ? numel4 : 1;
        const int blocks = (int)min((int64_t)((grid_work + threads - 1) / threads), (int64_t)1024);

        auto partial_sums = torch::empty({blocks}, options);

        mse_kernel_vec4<<<blocks, threads, threads * sizeof(float)>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            partial_sums.data_ptr<float>(),
            numel4,
            numel
        );

        if (blocks > 1) {
            reduce_partials_kernel<<<1, threads, threads * sizeof(float)>>>(
                partial_sums.data_ptr<float>(),
                output.data_ptr<float>(),
                blocks
            );
        } else {
            output.copy_(partial_sums);
        }
    } else {
        const int blocks = (int)min((int64_t)((numel + threads - 1) / threads), (int64_t)1024);
        auto partial_sums = torch::empty({blocks}, options);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            predictions.scalar_type(), "mse_kernel_partial", [&] {
                mse_kernel_partial<scalar_t><<<blocks, threads, threads * sizeof(float)>>>(
                    predictions.data_ptr<scalar_t>(),
                    targets.data_ptr<scalar_t>(),
                    partial_sums.data_ptr<float>(),
                    numel
                );
            }
        );

        if (blocks > 1) {
            reduce_partials_kernel<<<1, threads, threads * sizeof(float)>>>(
                partial_sums.data_ptr<float>(),
                output.data_ptr<float>(),
                blocks
            );
        } else {
            output.copy_(partial_sums);
        }
    }

    return output[0] / static_cast<float>(numel);
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that computes the Mean Squared Error loss for regression tasks.

        Parameters:
            None
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, predictions, targets):
        # <<<IMPROVE:forward_stmt_1>>>
        if (predictions.is_cuda and targets.is_cuda and
            predictions.is_contiguous() and targets.is_contiguous() and
            predictions.shape == targets.shape and
            predictions.dtype == targets.dtype and
            predictions.dtype in [torch.float32, torch.float16, torch.bfloat16]):
            return _stark_get_extension().fused_mse_loss(predictions, targets)
        else:
            return torch.mean((predictions - targets) ** 2)
        # <<<END_IMPROVE>>>
