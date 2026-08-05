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
    return f'stark_cuda_l2_p91_{digest}'

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

torch::Tensor convtranspose_epilogue_cuda(torch::Tensor x, torch::Tensor bias, double scaling_factor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convtranspose_epilogue_cuda", &convtranspose_epilogue_cuda, "Fused softmax+bias+scale+sigmoid epilogue (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

static inline __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

static inline __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 64-thread kernel: each thread owns channels tid and tid+64 (total 128 channels)
// Uses 2 warps; inter-warp combine done by thread 0 only (scalar, no warp reduction)
template <int C>
__global__ void fused_epilogue_c128_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N, const int H, const int W,
    const float scaling_factor
) {
    // shared[0..1]: per-warp max; shared[2]: block max
    // shared[3..4]: per-warp sum; shared[5]: block sum
    __shared__ float shared[6];

    const int spatial_idx = blockIdx.x;
    const int n = spatial_idx / (H * W);
    const int hw = spatial_idx % (H * W);
    const int h = hw / W;
    const int w = hw % W;
    if (n >= N) return;

    const int tid = threadIdx.x;   // 0..63
    const int lane = tid & 31;
    const int warp = tid >> 5;     // 0 or 1
    const int spatial_stride = H * W;
    const int base_offset = n * C * spatial_stride + h * W + w;

    // Each thread handles two channels: c0=tid, c1=tid+64
    const int c0 = tid;
    const int c1 = tid + 64;
    const float v0 = x[base_offset + c0 * spatial_stride];
    const float v1 = x[base_offset + c1 * spatial_stride];

    // Local max of the two owned channels
    float local_max = fmaxf(v0, v1);

    // Warp-level max reduction
    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) shared[warp] = warp_max;  // shared[0]=warp0 max, shared[1]=warp1 max
    __syncthreads();

    // Thread 0 combines the two warp maxes (scalar, no extra warp shuffle)
    if (tid == 0) shared[2] = fmaxf(shared[0], shared[1]);
    __syncthreads();
    const float block_max = shared[2];

    // Local sum of exps for two owned channels
    float e0 = __expf(v0 - block_max);
    float e1 = __expf(v1 - block_max);
    float local_sum = e0 + e1;

    // Warp-level sum reduction
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) shared[3 + warp] = warp_sum;  // shared[3]=warp0 sum, shared[4]=warp1 sum
    __syncthreads();

    // Thread 0 combines the two warp sums (scalar, no extra warp shuffle)
    if (tid == 0) shared[5] = shared[3] + shared[4];
    __syncthreads();
    const float block_sum = shared[5];

    // Write two outputs per thread
    const float inv_sum = 1.0f / block_sum;

    float s0 = e0 * inv_sum;
    float b0 = s0 + bias[c0];
    float sc0 = b0 * scaling_factor;
    out[base_offset + c0 * spatial_stride] = 1.0f / (1.0f + __expf(-sc0));

    float s1 = e1 * inv_sum;
    float b1 = s1 + bias[c1];
    float sc1 = b1 * scaling_factor;
    out[base_offset + c1 * spatial_stride] = 1.0f / (1.0f + __expf(-sc1));
}

template <int BLOCK_SIZE>
__global__ void fused_epilogue_generic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N, const int C, const int H, const int W,
    const float scaling_factor
) {
    extern __shared__ float shared_mem[];
    float* s_max = shared_mem;
    float* s_sum = shared_mem + BLOCK_SIZE;

    const int spatial_idx = blockIdx.x;
    const int n = spatial_idx / (H * W);
    const int hw = spatial_idx % (H * W);
    const int h = hw / W;
    const int w = hw % W;

    if (n >= N) return;

    const int tid = threadIdx.x;
    const int spatial_stride = H * W;
    const int base_offset = n * C * spatial_stride + h * W + w;

    float thread_max = -INFINITY;
    for (int c = tid; c < C; c += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, x[base_offset + c * spatial_stride]);
    }

    s_max[tid] = thread_max;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    const float global_max = s_max[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int c = tid; c < C; c += BLOCK_SIZE) {
        thread_sum += __expf(x[base_offset + c * spatial_stride] - global_max);
    }

    s_sum[tid] = thread_sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    const float global_sum = s_sum[0];

    for (int c = tid; c < C; c += BLOCK_SIZE) {
        const float softmax_val = __expf(x[base_offset + c * spatial_stride] - global_max) / global_sum;
        const float biased = softmax_val + bias[c];
        const float scaled = biased * scaling_factor;
        out[base_offset + c * spatial_stride] = 1.0f / (1.0f + __expf(-scaled));
    }
}

torch::Tensor convtranspose_epilogue_cuda(torch::Tensor x, torch::Tensor bias, double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    TORCH_CHECK(bias.size(0) == C && bias.size(1) == 1 && bias.size(2) == 1, "bias shape must be [C,1,1]");

    auto out = torch::empty_like(x);
    const int spatial_size = N * H * W;
    const float scale = static_cast<float>(scaling_factor);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (C == 128) {
        // 64 threads per block, each thread handles 2 channels (c and c+64)
        fused_epilogue_c128_kernel<128><<<spatial_size, 64, 0, stream>>>(
            x.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            N, H, W,
            scale
        );
    } else {
        constexpr int BLOCK_SIZE = 256;
        const int shared_mem_size = 2 * BLOCK_SIZE * sizeof(float);
        fused_epilogue_generic_kernel<BLOCK_SIZE><<<spatial_size, BLOCK_SIZE, shared_mem_size, stream>>>(
            x.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, H, W,
            scale
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed convolution, applies softmax, adds a bias term, scales the result, and applies sigmoid.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 4 and x.is_contiguous() and self.bias.is_contiguous():
            x = _stark_get_extension().convtranspose_epilogue_cuda(x, self.bias, float(self.scaling_factor))
        else:
            x = torch.softmax(x, dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4 and x.is_contiguous() and self.bias.is_contiguous()):
            x = x + self.bias
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4 and x.is_contiguous() and self.bias.is_contiguous()):
            x = x * self.scaling_factor
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if not (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4 and x.is_contiguous() and self.bias.is_contiguous()):
            x = torch.sigmoid(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
