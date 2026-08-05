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
    return f'stark_cuda_l1_p96_{digest}'

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

torch::Tensor huber_loss_forward(torch::Tensor predictions, torch::Tensor targets);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("huber_loss_forward", &huber_loss_forward, "Fused Huber loss forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__device__ __forceinline__ float huber_val(float diff) {
    float abs_diff = fabsf(diff);
    return (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void __launch_bounds__(BLOCK_SIZE, 4)
huber_loss_kernel(const float* __restrict__ preds,
                  const float* __restrict__ tgts,
                  float* __restrict__ partial_sums,
                  int n4,
                  int n) {
    __shared__ float smem[BLOCK_SIZE / 32];

    float acc = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    const float4* preds4 = reinterpret_cast<const float4*>(preds);
    const float4* tgts4 = reinterpret_cast<const float4*>(tgts);
    for (int i = tid; i < n4; i += stride) {
        float4 p = __ldg(preds4 + i);
        float4 t = __ldg(tgts4 + i);
        acc += huber_val(p.x - t.x);
        acc += huber_val(p.y - t.y);
        acc += huber_val(p.z - t.z);
        acc += huber_val(p.w - t.w);
    }
    for (int i = n4 * 4 + tid; i < n; i += stride) {
        acc += huber_val(__ldg(preds + i) - __ldg(tgts + i));
    }

    acc = warp_reduce_sum(acc);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        acc = (lane < (BLOCK_SIZE / 32)) ? smem[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (lane == 0) partial_sums[blockIdx.x] = acc;
    }
}

__global__ void reduce_partials_kernel(float* partial_sums, float* out, int num_blocks, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < num_blocks; i += BLOCK_SIZE)
        acc += partial_sums[i];
    smem[tid] = acc;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile float* vs = smem;
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];
        vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];
        vs[tid] += vs[tid + 1];
    }
    if (tid == 0) out[0] = smem[0] / (float)n;
}

torch::Tensor huber_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "inputs must be float32");

    int n = predictions.numel();
    int n4 = n / 4;

    int blocks = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto partial_sums = torch::empty({blocks}, predictions.options());
    auto out = torch::empty({}, predictions.options());

    huber_loss_kernel<<<blocks, BLOCK_SIZE, 0, stream.stream()>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n4, n);

    reduce_partials_kernel<<<1, BLOCK_SIZE, 0, stream.stream()>>>(
        partial_sums.data_ptr<float>(),
        out.data_ptr<float>(),
        blocks, n);

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that computes Smooth L1 (Huber) Loss for regression tasks.

        Parameters:
            None
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, predictions, targets):
        # <<<IMPROVE:forward_stmt_1>>>
        if (predictions.is_cuda and targets.is_cuda and \
            predictions.is_contiguous() and targets.is_contiguous() and \
            predictions.dtype == torch.float32 and targets.dtype == torch.float32 and \
            predictions.shape == targets.shape):
            return _stark_get_extension().huber_loss_forward(predictions, targets)
        return torch.nn.functional.smooth_l1_loss(predictions, targets)
        # <<<END_IMPROVE>>>
