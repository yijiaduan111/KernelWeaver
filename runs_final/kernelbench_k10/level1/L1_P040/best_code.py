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
    return f'stark_cuda_l1_p40_{digest}'

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

torch::Tensor layernorm_tail_fp32_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps);

torch::Tensor layernorm_tail_fp32(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "weight must be contiguous CUDA tensor");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bias must be contiguous CUDA tensor");
    return layernorm_tail_fp32_cuda(x, weight, bias, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_tail_fp32", &layernorm_tail_fp32, "Fused LayerNorm for fp32 tail-normalized tensors");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

// Warp-level reduction helper
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Number of warps per block
#define NUM_WARPS (THREADS_PER_BLOCK / WARP_SIZE)

// Scalar fallback kernel: one block per outer sample
__global__ void layernorm_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int outer,
    int inner,
    float eps) {

    int sample = blockIdx.x;
    if (sample >= outer) return;

    const float* x_row = x + (long long)sample * inner;
    float* out_row = out + (long long)sample * inner;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int i = tid; i < inner; i += nthreads) {
        float val = x_row[i];
        local_sum += val;
        local_sumsq += val * val;
    }

    local_sum   = warp_reduce_sum(local_sum);
    local_sumsq = warp_reduce_sum(local_sumsq);

    __shared__ float smem_sum[NUM_WARPS];
    __shared__ float smem_sumsq[NUM_WARPS];

    if (lane == 0) {
        smem_sum[warp_id]   = local_sum;
        smem_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float s  = (lane < NUM_WARPS) ? smem_sum[lane]   : 0.0f;
        float sq = (lane < NUM_WARPS) ? smem_sumsq[lane] : 0.0f;
        s  = warp_reduce_sum(s);
        sq = warp_reduce_sum(sq);
        if (lane == 0) {
            smem_sum[0]   = s;
            smem_sumsq[0] = sq;
        }
    }
    __syncthreads();

    __shared__ float mean_sh;
    __shared__ float invstd_sh;
    if (tid == 0) {
        float mean = smem_sum[0] / (float)inner;
        float var  = smem_sumsq[0] / (float)inner - mean * mean;
        mean_sh   = mean;
        invstd_sh = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean   = mean_sh;
    float invstd = invstd_sh;

    for (int i = tid; i < inner; i += nthreads) {
        float val = (x_row[i] - mean) * invstd;
        out_row[i] = val * weight[i] + bias[i];
    }
}

// Vectorized kernel using float4 with unroll-by-2: one block per outer sample
__global__ void layernorm_fused_kernel_vec4(
    const float4* __restrict__ x4,
    const float4* __restrict__ weight4,
    const float4* __restrict__ bias4,
    float4* __restrict__ out4,
    int outer,
    int inner4,  // inner / 4
    float eps) {

    int sample = blockIdx.x;
    if (sample >= outer) return;

    const float4* x_row = x4 + (long long)sample * inner4;
    float4* out_row = out4 + (long long)sample * inner4;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    // Unroll-by-2 stats loop
    int stride2 = nthreads * 2;
    int base = tid * 2;
    for (int i = base; i + 1 < inner4; i += stride2) {
        float4 vx0 = x_row[i];
        float4 vx1 = x_row[i + 1];
        local_sum   += vx0.x + vx0.y + vx0.z + vx0.w
                     + vx1.x + vx1.y + vx1.z + vx1.w;
        local_sumsq += vx0.x*vx0.x + vx0.y*vx0.y + vx0.z*vx0.z + vx0.w*vx0.w
                     + vx1.x*vx1.x + vx1.y*vx1.y + vx1.z*vx1.z + vx1.w*vx1.w;
    }
    // Cleanup: handle the tail element if inner4 is odd relative to stride2
    {
        // Each thread's second element index in the last iteration
        // We need to cover any index that was skipped: iterate scalar over remainder
        // Compute the start of the remainder region
        int full_iters = inner4 / stride2;  // number of complete unrolled iterations
        int covered = full_iters * stride2;  // elements covered by unrolled loop
        for (int i = covered + tid; i < inner4; i += nthreads) {
            float4 vx = x_row[i];
            local_sum   += vx.x + vx.y + vx.z + vx.w;
            local_sumsq += vx.x*vx.x + vx.y*vx.y + vx.z*vx.z + vx.w*vx.w;
        }
    }

    local_sum   = warp_reduce_sum(local_sum);
    local_sumsq = warp_reduce_sum(local_sumsq);

    __shared__ float smem_sum[NUM_WARPS];
    __shared__ float smem_sumsq[NUM_WARPS];

    if (lane == 0) {
        smem_sum[warp_id]   = local_sum;
        smem_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float s  = (lane < NUM_WARPS) ? smem_sum[lane]   : 0.0f;
        float sq = (lane < NUM_WARPS) ? smem_sumsq[lane] : 0.0f;
        s  = warp_reduce_sum(s);
        sq = warp_reduce_sum(sq);
        if (lane == 0) {
            smem_sum[0]   = s;
            smem_sumsq[0] = sq;
        }
    }
    __syncthreads();

    __shared__ float mean_sh;
    __shared__ float invstd_sh;
    if (tid == 0) {
        int inner = inner4 * 4;
        float mean = smem_sum[0] / (float)inner;
        float var  = smem_sumsq[0] / (float)inner - mean * mean;
        mean_sh   = mean;
        invstd_sh = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean   = mean_sh;
    float invstd = invstd_sh;

    // Unroll-by-2 writeback loop
    for (int i = base; i + 1 < inner4; i += stride2) {
        float4 vx0 = x_row[i];
        float4 vw0 = weight4[i];
        float4 vb0 = bias4[i];
        float4 vo0;
        vo0.x = ((vx0.x - mean) * invstd) * vw0.x + vb0.x;
        vo0.y = ((vx0.y - mean) * invstd) * vw0.y + vb0.y;
        vo0.z = ((vx0.z - mean) * invstd) * vw0.z + vb0.z;
        vo0.w = ((vx0.w - mean) * invstd) * vw0.w + vb0.w;
        out_row[i] = vo0;

        float4 vx1 = x_row[i + 1];
        float4 vw1 = weight4[i + 1];
        float4 vb1 = bias4[i + 1];
        float4 vo1;
        vo1.x = ((vx1.x - mean) * invstd) * vw1.x + vb1.x;
        vo1.y = ((vx1.y - mean) * invstd) * vw1.y + vb1.y;
        vo1.z = ((vx1.z - mean) * invstd) * vw1.z + vb1.z;
        vo1.w = ((vx1.w - mean) * invstd) * vw1.w + vb1.w;
        out_row[i + 1] = vo1;
    }
    // Cleanup writeback
    {
        int full_iters = inner4 / stride2;
        int covered = full_iters * stride2;
        for (int i = covered + tid; i < inner4; i += nthreads) {
            float4 vx = x_row[i];
            float4 vw = weight4[i];
            float4 vb = bias4[i];
            float4 vo;
            vo.x = ((vx.x - mean) * invstd) * vw.x + vb.x;
            vo.y = ((vx.y - mean) * invstd) * vw.y + vb.y;
            vo.z = ((vx.z - mean) * invstd) * vw.z + vb.z;
            vo.w = ((vx.w - mean) * invstd) * vw.w + vb.w;
            out_row[i] = vo;
        }
    }
}

torch::Tensor layernorm_tail_fp32_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {

    auto out = torch::empty_like(x);
    int outer = (int)x.size(0);
    long long inner = x.numel() / outer;

    dim3 grid(outer);
    dim3 block(THREADS_PER_BLOCK);

    bool can_vec = (inner % 4 == 0) &&
                   ((reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16) == 0) &&
                   ((reinterpret_cast<uintptr_t>(weight.data_ptr<float>()) % 16) == 0) &&
                   ((reinterpret_cast<uintptr_t>(bias.data_ptr<float>()) % 16) == 0) &&
                   ((reinterpret_cast<uintptr_t>(out.data_ptr<float>()) % 16) == 0);

    if (can_vec) {
        int inner4 = (int)(inner / 4);
        layernorm_fused_kernel_vec4<<<grid, block>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<const float4*>(weight.data_ptr<float>()),
            reinterpret_cast<const float4*>(bias.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            outer,
            inner4,
            (float)eps
        );
    } else {
        layernorm_fused_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            outer,
            (int)inner,
            (float)eps
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Layer Normalization.
        """
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the LayerNorm layer.

                Args:
                    normalized_shape (tuple): Shape of the input tensor to be normalized.
                """
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if (
            x.is_cuda and
            x.dtype == torch.float32 and
            x.is_contiguous() and
            x.dim() == 4 and
            tuple(x.shape[1:]) == (64, 256, 256) and
            self.ln.weight is not None and
            self.ln.bias is not None and
            self.ln.weight.is_cuda and
            self.ln.bias.is_cuda and
            self.ln.weight.is_contiguous() and
            self.ln.bias.is_contiguous() and
            tuple(self.ln.normalized_shape) == (64, 256, 256)
        ):
            return _stark_get_extension().layernorm_tail_fp32(
                x, self.ln.weight, self.ln.bias, float(self.ln.eps)
            )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return self.ln(x)
        # <<<END_IMPROVE>>>
