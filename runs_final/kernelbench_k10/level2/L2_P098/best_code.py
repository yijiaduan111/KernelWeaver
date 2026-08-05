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
    return f'stark_cuda_l2_p98_{digest}'

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

torch::Tensor fused_pool_gelu_scale_max(torch::Tensor x, double scale_factor, int64_t pool_kernel_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_pool_gelu_scale_max", &fused_pool_gelu_scale_max, "Fused AvgPool1d+GELU+Scale+Max (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define WARP_SIZE 32

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Specialized kernel for pool_kernel_size=16 with float4 vectorized loads
__global__ void fused_pool_gelu_scale_max_kernel_k16(
    const float* __restrict__ input,
    float* __restrict__ output,
    int features,
    float scale_factor,
    int batch_size
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_ptr = input + row * features;
    int num_pools = features / 16;

    float thread_max = -1e38f;

    // Each thread processes multiple pool windows using float4 loads (4 floats per load)
    // 16 floats per pool window = 4 float4 loads per window
    for (int pool_idx = threadIdx.x; pool_idx < num_pools; pool_idx += blockDim.x) {
        int base = pool_idx * 16;
        const float4* ptr4 = reinterpret_cast<const float4*>(row_ptr + base);
        float4 v0 = ptr4[0];
        float4 v1 = ptr4[1];
        float4 v2 = ptr4[2];
        float4 v3 = ptr4[3];

        float sum = v0.x + v0.y + v0.z + v0.w
                  + v1.x + v1.y + v1.z + v1.w
                  + v2.x + v2.y + v2.z + v2.w
                  + v3.x + v3.y + v3.z + v3.w;

        float pooled_val = sum * (1.0f / 16.0f);
        float val = gelu_exact(pooled_val) * scale_factor;
        thread_max = fmaxf(thread_max, val);
    }

    __shared__ float warp_maxes[32];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    float warp_max = warp_reduce_max(thread_max);

    if (lane == 0) {
        warp_maxes[warp_id] = warp_max;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float block_max = -1e38f;
        for (int i = 0; i < num_warps; i++) {
            block_max = fmaxf(block_max, warp_maxes[i]);
        }
        output[row] = block_max;
    }
}

// Generic kernel for arbitrary pool_kernel_size
__global__ void fused_pool_gelu_scale_max_kernel_generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    int features,
    int pool_kernel_size,
    float scale_factor,
    int batch_size
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_ptr = input + row * features;
    int num_pools = features / pool_kernel_size;

    float thread_max = -1e38f;

    for (int pool_idx = threadIdx.x; pool_idx < num_pools; pool_idx += blockDim.x) {
        int start_idx = pool_idx * pool_kernel_size;
        float sum = 0.0f;
        int end_idx = start_idx + pool_kernel_size;
        if (end_idx > features) end_idx = features;
        for (int k = start_idx; k < end_idx; k++) {
            sum += row_ptr[k];
        }
        float pooled_val = sum / (float)(end_idx - start_idx);
        float val = gelu_exact(pooled_val) * scale_factor;
        thread_max = fmaxf(thread_max, val);
    }

    __shared__ float warp_maxes[32];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    float warp_max = warp_reduce_max(thread_max);

    if (lane == 0) {
        warp_maxes[warp_id] = warp_max;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float block_max = -1e38f;
        for (int i = 0; i < num_warps; i++) {
            block_max = fmaxf(block_max, warp_maxes[i]);
        }
        output[row] = block_max;
    }
}

torch::Tensor fused_pool_gelu_scale_max(torch::Tensor x, double scale_factor, int64_t pool_kernel_size) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    x = x.contiguous();

    int batch_size = x.size(0);
    int features = x.size(1);

    auto output = torch::empty({batch_size}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (pool_kernel_size == 16 && (features % 16 == 0) && ((uintptr_t)x.data_ptr<float>() % 16 == 0)) {
        // Specialized float4 path for kernel_size=16
        const int threads = 256;
        fused_pool_gelu_scale_max_kernel_k16<<<batch_size, threads, 0, stream>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            features,
            (float)scale_factor,
            batch_size
        );
    } else {
        const int threads = 256;
        fused_pool_gelu_scale_max_kernel_generic<<<batch_size, threads, 0, stream>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            features,
            (int)pool_kernel_size,
            (float)scale_factor,
            batch_size
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
        """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

        orig_pool_kernel_size = pool_kernel_size
        if isinstance(orig_pool_kernel_size, (tuple, list)):
            orig_pool_kernel_size = orig_pool_kernel_size[0]

        if isinstance(orig_pool_kernel_size, int) and orig_pool_kernel_size > 0 and out_features % orig_pool_kernel_size == 0:
            pooled_out_features = out_features // orig_pool_kernel_size
            full_matmul = nn.Linear(in_features, out_features)
            has_bias = full_matmul.bias is not None
            pooled_matmul = nn.Linear(in_features, pooled_out_features, bias=has_bias)
            with torch.no_grad():
                pooled_matmul.weight.copy_(
                full_matmul.weight.view(pooled_out_features, orig_pool_kernel_size, in_features).mean(dim=1)
                )
                if has_bias:
                    pooled_matmul.bias.copy_(
                    full_matmul.bias.view(pooled_out_features, orig_pool_kernel_size).mean(dim=1)
                    )
            self.matmul = pooled_matmul
            # Use Identity so forward_stmt_2 sees a non-int kernel_size and takes the direct matmul path
            self.avg_pool = nn.Identity()
            self.avg_pool.kernel_size = None
        else:
            self.matmul = nn.Linear(in_features, out_features)
            self.avg_pool = nn.AvgPool1d(kernel_size=orig_pool_kernel_size)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_features).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_features).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        ks = self.avg_pool.kernel_size
        if isinstance(ks, (tuple, list)):
            ks = ks[0]

        used_weight_pool_fastpath = False
        if isinstance(ks, int) and ks > 0 and self.matmul.out_features % ks == 0:
            out_features = self.matmul.out_features
            in_features = self.matmul.in_features
            pooled_out = out_features // ks

            w = self.matmul.weight.view(pooled_out, ks, in_features).mean(dim=1)
            if self.matmul.bias is not None:
                b = self.matmul.bias.view(pooled_out, ks).mean(dim=1)
            else:
                b = None

            x = torch.nn.functional.linear(x, w, b)
            used_weight_pool_fastpath = True
        else:
            x = self.matmul(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if used_weight_pool_fastpath:
            x = x
        else:
            x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = torch.nn.functional.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = x * self.scale_factor
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        x = torch.max(x, dim=1).values
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
