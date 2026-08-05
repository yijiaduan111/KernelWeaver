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
    return f'stark_cuda_l2_p43_{digest}'

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
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# <<<END_IMPROVE>>>

CUDA_CPP_SRC = r"""
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>

torch::Tensor postpool_logsumexp_relu_cuda(torch::Tensor input);

torch::Tensor postpool_logsumexp_relu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (NCDHW)");
    return postpool_logsumexp_relu_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("postpool_logsumexp_relu", &postpool_logsumexp_relu, "Post-pool channel LogSumExp+ReLU (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Each block handles one spatial voxel (n, spatial) of an already max-pooled tensor.
// Reduces over channel dimension with stable logsumexp and applies ReLU to the scalar result.
__launch_bounds__(128, 4)
__global__ void postpool_logsumexp_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int DHW
) {
    extern __shared__ float smem[];

    const int voxel_idx = blockIdx.x;
    const int n       = voxel_idx / DHW;
    const int spatial = voxel_idx % DHW;

    if (n >= N) return;

    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    // smem layout: [0 .. C-1] channel values, [C .. C+bdim-1] per-thread reduction scratch
    float* reduce_buf = smem + C;

    // Phase 1: each thread loads its channels and computes thread-local max
    float local_max = -FLT_MAX;
    for (int c = tid; c < C; c += bdim) {
        float v = input[(n * C + c) * DHW + spatial];
        smem[c] = v;
        local_max = fmaxf(local_max, v);
    }
    reduce_buf[tid] = local_max;
    __syncthreads();

    // Block-level max reduction (bdim is a power of 2)
    for (int s = bdim >> 1; s >= 1; s >>= 1) {
        if (tid < s) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        }
        __syncthreads();
    }
    const float global_max = reduce_buf[0];
    __syncthreads();

    // Phase 2: each thread accumulates exp(val - global_max) for its channels
    float local_sum = 0.0f;
    for (int c = tid; c < C; c += bdim) {
        local_sum += expf(smem[c] - global_max);
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();

    // Block-level sum reduction
    for (int s = bdim >> 1; s >= 1; s >>= 1) {
        if (tid < s) {
            reduce_buf[tid] += reduce_buf[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float lse = global_max + logf(reduce_buf[0]);
        output[n * DHW + spatial] = fmaxf(lse, 0.0f);
    }
}

torch::Tensor postpool_logsumexp_relu_cuda(torch::Tensor input) {
    input = input.contiguous();
    const int N   = input.size(0);
    const int C   = input.size(1);
    const int D   = input.size(2);
    const int H   = input.size(3);
    const int W   = input.size(4);
    const int DHW = D * H * W;

    // Output: (N, 1, D, H, W) â channel dim reduced to 1
    auto output = torch::empty({N, 1, D, H, W}, input.options());

    const int total_voxels = N * DHW;

    // Block size: one thread per channel for C=64; must be power of 2
    int block_threads = 64;
    if (C <= 32)block_threads = 32;
    else if (C > 64)  block_threads = 128;

    // Shared memory: C floats for channel values + block_threads floats for reduction
    const int smem_bytes = (C + block_threads) * sizeof(float);

    postpool_logsumexp_relu_kernel<<<total_voxels, block_threads, smem_bytes>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, DHW
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x: Input tensor of shape (batch_size, in_channels, depth, height, width)
                Returns:
                    Output tensor of shape (batch_size, out_channels, depth', height', width')
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.max_pool(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = _stark_get_extension().postpool_logsumexp_relu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # ReLU is fused inside postpool_logsumexp_relu kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
