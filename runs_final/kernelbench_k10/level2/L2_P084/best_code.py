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
    return f'stark_cuda_l2_p84_{digest}'

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

torch::Tensor bn_scale_softmax_cuda(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor scale,
    double eps);

torch::Tensor bn_scale_softmax(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor scale,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    return bn_scale_softmax_cuda(x, bn_weight, bn_bias, running_mean, running_var, scale, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bn_scale_softmax", &bn_scale_softmax, "Fused BN+Scale+Softmax (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel: BN eval + scale + row-wise softmax over [N, C] float32 tensors.
// One block per row; threads cooperate over the C dimension.
// Supports scale of size 1 (scalar broadcast) or size C (per-feature).
__global__ void bn_scale_softmax_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int N,
    int C,
    float eps,
    int scale_size) {

    extern __shared__ float smem[];
    // smem layout: [C] for BN-normalized values, then reduction scratch
    // We use a two-pass approach: store intermediate per-element values in smem
    // then do reduction for max and sum.
    // For C=8192 and typical block sizes, use threads to tile over C.

    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * C;
    float* out_row = out + row * C;

    int tid = threadIdx.x;
    int blockDim_x = blockDim.x;

    // Shared memory: used for reduction (max, sum)
    // We store the BN+scale result back into output first, then do softmax in-place.
    // Two-pass softmax: pass1 = compute BN+scale -> write to out_row, find row max
    //                   pass2 = compute exp(x-max), accumulate sum
    //                   pass3 = normalize by sum

    // Pass 1: BN + scale, write to out, find per-thread max
    float thread_max = -1e38f;
    for (int c = tid; c < C; c += blockDim_x) {
        float mean = running_mean[c];
        float var  = running_var[c];
        float gamma = bn_weight[c];
        float beta  = bn_bias[c];
        float s = (scale_size == 1) ? scale[0] : scale[c];
        float val = ((x_row[c] - mean) * rsqrtf(var + eps)) * gamma + beta;
        val = val * s;
        out_row[c] = val;
        thread_max = fmaxf(thread_max, val);
    }

    // Block reduction for max using shared memory
    float* smax = smem;  // size = blockDim_x
    smax[tid] = thread_max;
    __syncthreads();
    for (int stride = blockDim_x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = smax[0];
    __syncthreads();

    // Pass 2: exp(x - max), find per-thread sum
    float* ssum = smem;  // reuse for sum
    float thread_sum = 0.0f;
    for (int c = tid; c < C; c += blockDim_x) {
        float val = expf(out_row[c] - row_max);
        out_row[c] = val;
        thread_sum += val;
    }

    ssum[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim_x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = ssum[0];
    __syncthreads();

    // Pass 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int c = tid; c < C; c += blockDim_x) {
        out_row[c] *= inv_sum;
    }
}

torch::Tensor bn_scale_softmax_cuda(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor scale,
    double eps) {

    int N = x.size(0);
    int C = x.size(1);
    int scale_size = (int)scale.numel();

    auto out = torch::empty_like(x);

    // Use 256 threads per block; each block handles one row.
    // For C=8192, each thread handles 32 elements.
    const int threads = 256;
    const int blocks = N;
    // Shared memory: threads floats for reduction
    size_t smem_bytes = threads * sizeof(float);

    bn_scale_softmax_kernel<<<blocks, threads, smem_bytes>>>(
        x.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C,
        (float)eps,
        scale_size
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
        """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Use fused CUDA path when: eval mode, CUDA float32 2D contiguous tensor.
        _use_fused = (
            not self.training
            and x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 2
            and x.is_contiguous()
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if _use_fused:
            return _stark_get_extension().bn_scale_softmax(
            x,
            self.bn.weight.contiguous(),
            self.bn.bias.contiguous(),
            self.bn.running_mean.contiguous(),
            self.bn.running_var.contiguous(),
            self.scale.contiguous(),
            self.bn.eps,
            )
        x = self.bn(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = self.scale * x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = self.softmax(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
