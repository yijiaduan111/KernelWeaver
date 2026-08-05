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
    return f'stark_cuda_l1_p100_{digest}'

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

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hinge_loss_cuda", &hinge_loss_cuda, "Fused hinge loss CUDA kernel");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Scalar fused hinge-loss reduction kernel
__global__ void hinge_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        float margin = 1.0f - predictions[i] * targets[i];
        local_sum += fmaxf(0.0f, margin);
    }

    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();

    int warps_per_block = (blockDim.x + 31) >> 5;
    if (warp_id == 0) {
        float val = (lane < warps_per_block) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane == 0) {
            atomicAdd(output, val);
        }
    }
}

// float4 vectorized fused hinge-loss reduction kernel
__global__ void hinge_loss_kernel_vec4(
    const float4* __restrict__ predictions,
    const float4* __restrict__ targets,
    float* __restrict__ output,
    int n_vec,   // number of float4 elements (n / 4)
    int n_tail,  // number of remaining scalar elements (n % 4)
    const float* __restrict__ pred_tail,
    const float* __restrict__ tgt_tail
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    // Vectorized main loop
    for (int i = idx; i < n_vec; i += stride) {
        float4 p = predictions[i];
        float4 t = targets[i];
        local_sum += fmaxf(0.0f, 1.0f - p.x * t.x);
        local_sum += fmaxf(0.0f, 1.0f - p.y * t.y);
        local_sum += fmaxf(0.0f, 1.0f - p.z * t.z);
        local_sum += fmaxf(0.0f, 1.0f - p.w * t.w);
    }

    // Thread 0 of block 0 handles the tail
    if (blockIdx.x == 0 && tid == 0) {
        for (int i = 0; i < n_tail; i++) {
            float margin = 1.0f - pred_tail[i] * tgt_tail[i];
            local_sum += fmaxf(0.0f, margin);
        }
    }

    // Warp reduction
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();

    int warps_per_block = (blockDim.x + 31) >> 5;
    if (warp_id == 0) {
        float val = (lane < warps_per_block) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane == 0) {
            atomicAdd(output, val);
        }
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    auto pred_c = predictions.contiguous().view({-1});
    auto tgt_c = targets.contiguous().view({-1});
    TORCH_CHECK(pred_c.numel() == tgt_c.numel(), "predictions and targets must have the same number of elements");

    int n = pred_c.numel();
    auto output = torch::zeros({1}, pred_c.options());

    const int block_size = 256;
    int warps_per_block = block_size / 32;
    int smem_bytes = warps_per_block * sizeof(float);

    const float* pred_ptr = pred_c.data_ptr<float>();
    const float* tgt_ptr  = tgt_c.data_ptr<float>();

    bool use_vec4 = (pred_c.scalar_type() == at::kFloat) &&
                    (tgt_c.scalar_type() == at::kFloat) &&
                    (n >= 4) &&
                    ((reinterpret_cast<uintptr_t>(pred_ptr) % 16) == 0) &&
                    ((reinterpret_cast<uintptr_t>(tgt_ptr)  % 16) == 0);

    if (use_vec4) {
        int n_vec  = n / 4;
        int n_tail = n % 4;
        int grid_size = (n_vec + block_size - 1) / block_size;
        grid_size = min(grid_size, 1024);
        // Ensure block 0 exists to handle tail
        if (grid_size == 0) grid_size = 1;

        hinge_loss_kernel_vec4<<<grid_size, block_size, smem_bytes>>>(
            reinterpret_cast<const float4*>(pred_ptr),
            reinterpret_cast<const float4*>(tgt_ptr),
            output.data_ptr<float>(),
            n_vec,
            n_tail,
            pred_ptr + n_vec * 4,
            tgt_ptr  + n_vec * 4
        );
    } else {
        int grid_size = (n + block_size - 1) / block_size;
        grid_size = min(grid_size, 1024);

        hinge_loss_kernel<<<grid_size, block_size, smem_bytes>>>(
            pred_ptr,
            tgt_ptr,
            output.data_ptr<float>(),
            n
        );
    }

    return (output / static_cast<float>(n)).squeeze();
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that computes Hinge Loss for binary classification tasks.

        Parameters:
            None
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, predictions, targets):
        # <<<IMPROVE:forward_stmt_1>>>
        predictions, targets = torch.broadcast_tensors(predictions, targets)
        return _stark_get_extension().hinge_loss_cuda(predictions, targets)
        # <<<END_IMPROVE>>>
