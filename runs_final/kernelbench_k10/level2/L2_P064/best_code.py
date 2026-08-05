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
    return f'stark_cuda_l2_p64_{digest}'

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

torch::Tensor fused_logsumexp_activations(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_logsumexp_activations", &fused_logsumexp_activations, "Fused rowwise logsumexp + activations (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32

__device__ __forceinline__ float leaky_relu_f(float x, float slope) {
    return x >= 0.0f ? x : slope * x;
}

__device__ __forceinline__ float gelu_f(float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ void merge_logsumexp(
    float& max_a, float& sum_a,
    float max_b, float sum_b)
{
    if (max_b > max_a) {
        sum_a = sum_a * expf(max_a - max_b) + sum_b;
        max_a = max_b;
    } else {
        sum_a = sum_a + sum_b * expf(max_b - max_a);
    }
}

// Specialized hot-path kernel for N==8192, blockDim=256
// Each block handles one row; 256 threads each handle exactly 8 float4 packs
// (2048 float4 total / 256 threads = 8 packs per thread)
__launch_bounds__(256, 4)
__global__ void fused_logsumexp_activations_kernel_8192(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const float4* row_ptr4 = reinterpret_cast<const float4*>(input + row * 8192);

    int base = threadIdx.x;

    float local_max = -1e38f;
    float local_sum = 0.0f;

    // Statically unrolled 8-load sequence: each thread reads indices base+k*256 for k=0..7
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float4 v = row_ptr4[base + k * 256];
        float chunk_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        if (chunk_max > local_max) {
            local_sum = local_sum * expf(local_max - chunk_max);
            local_max = chunk_max;
        }
        local_sum += expf(v.x - local_max) + expf(v.y - local_max)
                   + expf(v.z - local_max) + expf(v.w - local_max);
    }

    // Warp-level reduction
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(mask, local_max, offset);
        float other_sum = __shfl_xor_sync(mask, local_sum, offset);
        merge_logsumexp(local_max, local_sum, other_max, other_sum);
    }

    // Block-level reduction: 256 threads = 8 warps
    __shared__ float smem_max[8];
    __shared__ float smem_sum[8];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    if (lane == 0) {
        smem_max[wid] = local_max;
        smem_sum[wid] = local_sum;
    }
    __syncthreads();

    if (tid < 8) {
        local_max = smem_max[tid];
        local_sum = smem_sum[tid];
    } else {
        local_max = -1e38f;
        local_sum = 0.0f;
    }

    if (tid < WARP_SIZE) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_max = __shfl_xor_sync(mask, local_max, offset);
            float other_sum = __shfl_xor_sync(mask, local_sum, offset);
            merge_logsumexp(local_max, local_sum, other_max, other_sum);
        }
    }

    if (tid == 0) {
        float y = local_max + logf(local_sum);
        y = leaky_relu_f(y, 0.01f);
        y = leaky_relu_f(y, 0.01f);
        y = gelu_f(y);
        y = gelu_f(y);
        output[row] = y;
    }
}

// Generic fallback kernel for other row widths
__launch_bounds__(512, 2)
__global__ void fused_logsumexp_activations_kernel_generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_ptr = input + row * N;

    float local_max = -1e38f;
    float local_sum = 0.0f;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    bool aligned = ((uintptr_t)row_ptr % 16 == 0) && (N % 4 == 0);

    if (aligned) {
        const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
        int N4 = N / 4;
        for (int i = tid; i < N4; i += block_size) {
            float4 v = row_ptr4[i];
            float chunk_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
            if (chunk_max > local_max) {
                local_sum = local_sum * expf(local_max - chunk_max);
                local_max = chunk_max;
            }
            local_sum += expf(v.x - local_max) + expf(v.y - local_max)
                       + expf(v.z - local_max) + expf(v.w - local_max);
        }
    } else {
        for (int i = tid; i < N; i += block_size) {
            float v = row_ptr[i];
            if (v > local_max) {
                local_sum = local_sum * expf(local_max - v);
                local_max = v;
            } else {
                local_sum += expf(v - local_max);
            }
        }
    }

    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(mask, local_max, offset);
        float other_sum = __shfl_xor_sync(mask, local_sum, offset);
        merge_logsumexp(local_max, local_sum, other_max, other_sum);
    }

    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem + num_warps;

    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;

    if (lane == 0) {
        smem_max[wid] = local_max;
        smem_sum[wid] = local_sum;
    }
    __syncthreads();

    if (tid < num_warps) {
        local_max = smem_max[tid];
        local_sum = smem_sum[tid];
    } else {
        local_max = -1e38f;
        local_sum = 0.0f;
    }

    if (tid < WARP_SIZE) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_max = __shfl_xor_sync(mask, local_max, offset);
            float other_sum = __shfl_xor_sync(mask, local_sum, offset);
            merge_logsumexp(local_max, local_sum, other_max, other_sum);
        }
    }

    if (tid == 0) {
        float y = local_max + logf(local_sum);
        y = leaky_relu_f(y, 0.01f);
        y = leaky_relu_f(y, 0.01f);
        y = gelu_f(y);
        y = gelu_f(y);
        output[row] = y;
    }
}

torch::Tensor fused_logsumexp_activations(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int M = x.size(0);
    int N = x.size(1);

    auto output = torch::empty({M, 1}, x.options());

    if (N == 8192) {
        // Specialized path: 256 threads, fixed shared memory (8 warps), fully unrolled 8-pack loop
        fused_logsumexp_activations_kernel_8192<<<M, 256>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            M
        );
    } else {
        int block_size = 512;
        if (N <= 256) block_size = 128;
        else if (N <= 1024) block_size = 256;

        int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
        int smem_bytes = 2 * num_warps * sizeof(float);

        fused_logsumexp_activations_kernel_generic<<<M, block_size, smem_bytes>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
        LeakyReLU, GELU, and GELU activations.
        """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.linear(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous():
            x = _stark_get_extension().fused_logsumexp_activations(x)
        else:
            x = torch.logsumexp(x, dim=1, keepdim=True)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # leaky_relu fused into forward_stmt_2 CUDA path (fallback handled above)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # leaky_relu fused into forward_stmt_2 CUDA path (fallback handled above)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # gelu fused into forward_stmt_2 CUDA path (fallback handled above)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # gelu fused into forward_stmt_2 CUDA path (fallback handled above)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
