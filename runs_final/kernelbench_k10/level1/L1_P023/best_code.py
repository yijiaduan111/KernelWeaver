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
    return f'stark_cuda_l1_p23_{digest}'

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

torch::Tensor softmax_rowwise_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_rowwise_cuda", &softmax_rowwise_cuda, "Row-wise online softmax (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define THREADS 1024
#define NWARPS (THREADS / WARP_SIZE)

// Online softmax kernel: one block per row, 1024 threads
// Warp-level pair merge with shuffles; block-level merge via warp 0 shuffles.
__global__ void softmax_rowwise_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int rows, int cols) {

    // smem layout: [0..NWARPS) = warp max, [NWARPS..2*NWARPS) = warp sum
    __shared__ float smem[2 * NWARPS];

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    const float* row_in = input + (long long)row * cols;
    float* row_out = output + (long long)row * cols;

    // --- Pass 1: online (max, sum) accumulation ---
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    int cols4 = cols / 4;
    const float4* row_in4 = reinterpret_cast<const float4*>(row_in);

    for (int i = tid; i < cols4; i += THREADS) {
        float4 v = __ldg(row_in4 + i);
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float x = vals[k];
            float new_max = fmaxf(local_max, x);
            local_sum = local_sum * expf(local_max - new_max) + expf(x - new_max);
            local_max = new_max;
        }
    }
    // Handle remainder elements
    int base = cols4 * 4;
    for (int i = base + tid; i < cols; i += THREADS) {
        float x = __ldg(row_in + i);
        float new_max = fmaxf(local_max, x);
        local_sum = local_sum * expf(local_max - new_max) + expf(x - new_max);
        local_max = new_max;
    }

    // Warp-level reduction: combine (max, sum) pairs via shuffles
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(FULL_MASK, local_max, offset);
        float other_sum = __shfl_down_sync(FULL_MASK, local_sum, offset);
        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - new_max) + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        smem[warp_id] = local_max;
        smem[NWARPS + warp_id] = local_sum;
    }
    __syncthreads();

    // Block-level reduction: warp 0 loads warp summaries and reduces with shuffles
    float global_max, global_sum;
    if (warp_id == 0) {
        // Each lane in warp 0 loads one warp's (max, sum)
        float w_max = (lane_id < NWARPS) ? smem[lane_id] : -FLT_MAX;
        float w_sum = (lane_id < NWARPS) ? smem[NWARPS + lane_id] : 0.0f;

        // Reduce NWARPS=32 values with a full warp shuffle
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(FULL_MASK, w_max, offset);
            float other_sum = __shfl_down_sync(FULL_MASK, w_sum, offset);
            float new_max = fmaxf(w_max, other_max);
            w_sum = w_sum * expf(w_max - new_max) + other_sum * expf(other_max - new_max);
            w_max = new_max;
        }

        if (lane_id == 0) {
            smem[0] = w_max;
            smem[1] = w_sum;
        }
    }
    __syncthreads();

    global_max = smem[0];
    global_sum = smem[1];
    float inv_sum = 1.0f / global_sum;

    // --- Pass 2: write normalized output ---
    float4* row_out4 = reinterpret_cast<float4*>(row_out);
    for (int i = tid; i < cols4; i += THREADS) {
        float4 v = __ldg(row_in4 + i);
        float4 out;
        out.x = expf(v.x - global_max) * inv_sum;
        out.y = expf(v.y - global_max) * inv_sum;
        out.z = expf(v.z - global_max) * inv_sum;
        out.w = expf(v.w - global_max) * inv_sum;
        row_out4[i] = out;
    }
    for (int i = base + tid; i < cols; i += THREADS) {
        row_out[i] = expf(__ldg(row_in + i) - global_max) * inv_sum;
    }
}

torch::Tensor softmax_rowwise_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty_like(x);

    constexpr int threads = THREADS;
    constexpr int nwarps = NWARPS;
    size_t smem_size = 2 * nwarps * sizeof(float);

    softmax_rowwise_kernel<<<rows, threads, smem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a Softmax activation.
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
                Applies Softmax activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, num_features).

                Returns:
                    torch.Tensor: Output tensor with Softmax applied, same shape as input.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous():
            return _stark_get_extension().softmax_rowwise_cuda(x)
        return torch.softmax(x, dim=1)
        # <<<END_IMPROVE>>>
