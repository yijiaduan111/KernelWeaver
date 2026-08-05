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
    return f'stark_cuda_l1_p98_{digest}'

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

torch::Tensor kl_div_forward(torch::Tensor predictions, torch::Tensor targets);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kl_div_forward", &kl_div_forward, "Fused KL divergence forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int N
) {
    extern __shared__ float sdata[];

    float partial = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        float t = targets[i];
        float p = predictions[i];
        if (t > 0.0f) {
            partial += t * (logf(t) - logf(fmaxf(p, 1e-10f)));
        }
    }

    // Warp-level reduction
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(mask, partial, offset);
    }

    // Store warp results in shared memory
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        sdata[warp_id] = partial;
    }
    __syncthreads();

    // Block-level reduction over warp results
    int num_warps = blockDim.x >> 5;
    if (threadIdx.x < num_warps) {
        float val = sdata[threadIdx.x];
        for (int offset = num_warps >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, val);
        }
    }
}

torch::Tensor kl_div_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    auto pred_c = predictions.contiguous();
    auto tgt_c = targets.contiguous();

    int N = (int)pred_c.numel();
    int batch_size = (int)pred_c.size(0);

    auto output = torch::zeros({1}, pred_c.options());

    const int block_size = 256;
    int num_warps = block_size / 32;
    int grid_size = std::min((N + block_size - 1) / block_size, 2048);
    int smem = num_warps * sizeof(float);

    kl_div_kernel<<<grid_size, block_size, smem>>>(
        pred_c.data_ptr<float>(),
        tgt_c.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    return output / (float)batch_size;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A model that computes Kullback-Leibler Divergence for comparing two distributions.

        Parameters:
            None
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, predictions, targets):
        # <<<IMPROVE:forward_stmt_1>>>
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        result = _stark_get_extension().kl_div_forward(predictions, targets)
        return result.squeeze()
        # <<<END_IMPROVE>>>
