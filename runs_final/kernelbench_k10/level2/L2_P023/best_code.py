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
    return f'stark_cuda_l2_p23_{digest}'

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

torch::Tensor groupnorm_mean_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("groupnorm_mean_cuda", &groupnorm_mean_cuda,
          "Fused GroupNorm + global mean (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Specialized kernel for group_size == 3, contiguous NCDHW input.
// Hoists per-group base offsets and uses direct increments for 3-channel loads.
__global__ void groupnorm_mean_kernel_gs3(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int spatial,
    int num_groups,
    float eps
) {
    int ng   = blockIdx.x;
    int n    = ng / num_groups;
    int g    = ng % num_groups;
    int tid  = threadIdx.x;
    int bdim = blockDim.x;

    // Precompute base offset for channel g*3 once per block
    // group_base = n*C*spatial + (g*3)*spatial
    int64_t group_base = (int64_t)n * C * spatial + (int64_t)(g * 3) * spatial;
    // The three channel slices are at group_base, group_base+spatial, group_base+2*spatial
    const float* c0_ptr = x + group_base;
    const float* c1_ptr = c0_ptr + spatial;
    const float* c2_ptr = c1_ptr + spatial;

    int elems_per_group = 3 * spatial;

    float sum = 0.0f, sumsq = 0.0f;
    for (int s = tid; s < spatial; s += bdim) {
        float v0 = __ldg(c0_ptr + s);
        float v1 = __ldg(c1_ptr + s);
        float v2 = __ldg(c2_ptr + s);
        sum   += v0 + v1 + v2;
        sumsq += v0*v0 + v1*v1 + v2*v2;
    }

    extern __shared__ float smem[];
    int nwarps = (bdim + 31) / 32;
    float* smem_sum   = smem;
    float* smem_sumsq = smem + nwarps;
    int lane = tid & 31;
    int wid  = tid >> 5;

    sum   = warpReduceSum(sum);
    sumsq = warpReduceSum(sumsq);
    if (lane == 0) {
        smem_sum[wid]   = sum;
        smem_sumsq[wid] = sumsq;
    }
    __syncthreads();

    if (wid == 0) {
        sum   = (lane < nwarps) ? smem_sum[lane]   : 0.0f;
        sumsq = (lane < nwarps) ? smem_sumsq[lane] : 0.0f;
        sum   = warpReduceSum(sum);
        sumsq = warpReduceSum(sumsq);
        if (lane == 0) {
            smem_sum[0]   = sum;
            smem_sumsq[0] = sumsq;
        }
    }
    __syncthreads();

    float mean_val = smem_sum[0] * (1.0f / (float)elems_per_group);
    float var_val  = smem_sumsq[0] * (1.0f / (float)elems_per_group) - mean_val * mean_val;
    float inv_std  = rsqrtf(var_val + eps);

    // Preload affine params for all 3 channels into registers
    int c0 = g * 3;
    float w0 = weight[c0], w1 = weight[c0+1], w2 = weight[c0+2];
    float b0 = bias[c0],   b1 = bias[c0+1],   b2 = bias[c0+2];

    float acc = 0.0f;
    for (int s = tid; s < spatial; s += bdim) {
        float v0 = (__ldg(c0_ptr + s) - mean_val) * inv_std;
        float v1 = (__ldg(c1_ptr + s) - mean_val) * inv_std;
        float v2 = (__ldg(c2_ptr + s) - mean_val) * inv_std;
        acc += w0*v0 + b0 + w1*v1 + b1 + w2*v2 + b2;
    }

    __syncthreads();
    acc = warpReduceSum(acc);
    if (lane == 0) smem_sum[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        acc = (lane < nwarps) ? smem_sum[lane] : 0.0f;
        acc = warpReduceSum(acc);
        if (lane == 0) smem_sum[0] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(&out[n], smem_sum[0]);
    }
}

// Generic strided kernel (fallback for arbitrary shapes/layouts)
__global__ void groupnorm_mean_kernel_strided(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int D, int H, int W,
    int num_groups,
    float eps,
    int group_size,
    int64_t stride_n, int64_t stride_c, int64_t stride_d, int64_t stride_h, int64_t stride_w
) {
    int ng   = blockIdx.x;
    int n    = ng / num_groups;
    int g    = ng % num_groups;
    int tid  = threadIdx.x;
    int bdim = blockDim.x;

    int spatial = D * H * W;
    int elems_per_group = group_size * spatial;

    float sum = 0.0f, sumsq = 0.0f;
    for (int i = tid; i < elems_per_group; i += bdim) {
        int c_local = i / spatial;
        int spatial_idx = i % spatial;
        int d_idx = spatial_idx / (H * W);
        int hw_idx = spatial_idx % (H * W);
        int h_idx = hw_idx / W;
        int w_idx = hw_idx % W;
        int c_global = g * group_size + c_local;
        int64_t offset = n * stride_n + c_global * stride_c + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w;
        float v = __ldg(x + offset);
        sum   += v;
        sumsq += v * v;
    }

    extern __shared__ float smem[];
    int nwarps = (bdim + 31) / 32;
    float* smem_sum   = smem;
    float* smem_sumsq = smem + nwarps;
    int lane = tid & 31;
    int wid  = tid >> 5;

    sum   = warpReduceSum(sum);
    sumsq = warpReduceSum(sumsq);
    if (lane == 0) {
        smem_sum[wid]   = sum;
        smem_sumsq[wid] = sumsq;
    }
    __syncthreads();

    if (wid == 0) {
        sum   = (lane < nwarps) ? smem_sum[lane]   : 0.0f;
        sumsq = (lane < nwarps) ? smem_sumsq[lane] : 0.0f;
        sum   = warpReduceSum(sum);
        sumsq = warpReduceSum(sumsq);
        if (lane == 0) {
            smem_sum[0]   = sum;
            smem_sumsq[0] = sumsq;
        }
    }
    __syncthreads();

    float mean_val = smem_sum[0] * (1.0f / (float)elems_per_group);
    float var_val  = smem_sumsq[0] * (1.0f / (float)elems_per_group) - mean_val * mean_val;
    float inv_std  = rsqrtf(var_val + eps);

    float acc = 0.0f;
    for (int i = tid; i < elems_per_group; i += bdim) {
        int c_local = i / spatial;
        int spatial_idx = i % spatial;
        int d_idx = spatial_idx / (H * W);
        int hw_idx = spatial_idx % (H * W);
        int h_idx = hw_idx / W;
        int w_idx = hw_idx % W;
        int c_global = g * group_size + c_local;
        int64_t offset = n * stride_n + c_global * stride_c + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w;
        float v = (__ldg(x + offset) - mean_val) * inv_std;
        acc += weight[c_global] * v + bias[c_global];
    }

    __syncthreads();
    acc = warpReduceSum(acc);
    if (lane == 0) smem_sum[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        acc = (lane < nwarps) ? smem_sum[lane] : 0.0f;
        acc = warpReduceSum(acc);
        if (lane == 0) smem_sum[0] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(&out[n], smem_sum[0]);
    }
}

torch::Tensor groupnorm_mean_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,C,D,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int spatial = D * H * W;
    int group_size = C / (int)num_groups;
    int elems_per_group = group_size * spatial;

    auto out = torch::zeros({N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (group_size == 3 && x.is_contiguous()) {
        // Fast path: hoisted base offsets, no integer division per element
        int block_size = 256;
        if (spatial < 256) block_size = 128;
        if (spatial < 128) block_size = 64;
        int nwarps = (block_size + 31) / 32;
        size_t smem_bytes = (2 * nwarps + 1) * sizeof(float);
        int grid = N * (int)num_groups;
        groupnorm_mean_kernel_gs3<<<grid, block_size, smem_bytes, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, spatial,
            (int)num_groups,
            (float)eps
        );
    } else {
        // Generic strided fallback
        int block_size = 256;
        if (elems_per_group < 256) block_size = 128;
        if (elems_per_group < 128) block_size = 64;
        int nwarps = (block_size + 31) / 32;
        size_t smem_bytes = (2 * nwarps + 1) * sizeof(float);
        int grid = N * (int)num_groups;
        int64_t stride_n = x.stride(0);
        int64_t stride_c = x.stride(1);
        int64_t stride_d = x.stride(2);
        int64_t stride_h = x.stride(3);
        int64_t stride_w = x.stride(4);
        groupnorm_mean_kernel_strided<<<grid, block_size, smem_bytes, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, D, H, W,
            (int)num_groups,
            (float)eps,
            group_size,
            stride_n, stride_c, stride_d, stride_h, stride_w
        );
    }

    out.div_((float)(C * spatial));
    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D convolution, applies Group Normalization, computes the mean
        """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, 1).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if x.is_cuda and x.dtype == torch.float32:
            x = _stark_get_extension().groupnorm_mean_cuda(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps
            )
            _fused = True
        else:
            x = self.group_norm(x)
            _fused = False
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not _fused:
                    x = x.mean(dim=[1, 2, 3, 4])
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
