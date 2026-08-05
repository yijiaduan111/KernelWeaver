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
    return f'stark_cuda_l2_p77_{digest}'

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

torch::Tensor fused_scale_bn_globalavgpool_cuda(
    torch::Tensor input,
    float scale_factor,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    float momentum,
    bool training
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scale_bn_globalavgpool",
          &fused_scale_bn_globalavgpool_cuda,
          "Fused scale + batch norm + global avg pool (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void compute_stats_and_nc_sum_kernel(
    const float* __restrict__ input,
    float scale_factor,
    float* __restrict__ nc_sum_out,
    float* __restrict__ sum_out,
    float* __restrict__ sumsq_out,
    int N, int C, int DHW
) {
    int c = blockIdx.x;
    int T = blockDim.x;

    __shared__ float smem[32];
    __shared__ float smem_sum[32];
    __shared__ float smem_sumsq[32];

    float ch_sum = 0.0f;
    float ch_sumsq = 0.0f;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = (T + 31) >> 5;

    for (int n = 0; n < N; ++n) {
        const float* row = input + (n * C + c) * DHW;
        float n_local = 0.0f;

        int dhw = threadIdx.x;
        int stride = T;
        int limit4 = DHW - 4 * stride;
        for (; dhw <= limit4; dhw += 4 * stride) {
            float v0 = row[dhw] * scale_factor;
            float v1 = row[dhw + stride] * scale_factor;
            float v2 = row[dhw + 2 * stride] * scale_factor;
            float v3 = row[dhw + 3 * stride] * scale_factor;
            n_local += v0 + v1 + v2 + v3;
            ch_sum += v0 + v1 + v2 + v3;
            ch_sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        }
        for (; dhw < DHW; dhw += stride) {
            float v = row[dhw] * scale_factor;
            n_local += v;
            ch_sum += v;
            ch_sumsq += v * v;
        }

        n_local = warp_reduce_sum(n_local);
        if (lane == 0) {
            smem[wid] = n_local;
        }
        __syncthreads();

        float block_sum = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
        if (wid == 0) {
            block_sum = warp_reduce_sum(block_sum);
        }
        if (threadIdx.x == 0) {
            nc_sum_out[n * C + c] = block_sum;
        }
        __syncthreads();
    }

    ch_sum = warp_reduce_sum(ch_sum);
    ch_sumsq = warp_reduce_sum(ch_sumsq);
    if (lane == 0) {
        smem_sum[wid] = ch_sum;
        smem_sumsq[wid] = ch_sumsq;
    }
    __syncthreads();

    ch_sum = (threadIdx.x < nwarps) ? smem_sum[threadIdx.x] : 0.0f;
    ch_sumsq = (threadIdx.x < nwarps) ? smem_sumsq[threadIdx.x] : 0.0f;
    if (wid == 0) {
        ch_sum = warp_reduce_sum(ch_sum);
        ch_sumsq = warp_reduce_sum(ch_sumsq);
    }
    if (threadIdx.x == 0) {
        sum_out[c] = ch_sum;
        sumsq_out[c] = ch_sumsq;
    }
}

__global__ void compute_output_from_sums_kernel(
    const float* __restrict__ nc_sum,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N, int C, float inv_DHW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;
    int n = idx / C;
    int c = idx % C;
    float nc_mean = nc_sum[n * C + c] * inv_DHW;
    float normed = (nc_mean - mean[c]) * inv_std[c];
    out[idx] = gamma[c] * normed + beta[c];
}

__global__ void eval_affine_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ a_coeff,
    const float* __restrict__ b_coeff,
    float* __restrict__ out,
    int N, int C, int DHW
) {
    int c = blockIdx.x;
    int n = blockIdx.y;
    float inv_DHW = 1.0f / (float)DHW;

    float local_sum = 0.0f;
    const float* row = input + (n * C + c) * DHW;
    for (int dhw = threadIdx.x; dhw < DHW; dhw += blockDim.x) {
        local_sum += row[dhw];
    }

    local_sum = warp_reduce_sum(local_sum);
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) {
        smem[wid] = local_sum;
    }
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    local_sum = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) {
        local_sum = warp_reduce_sum(local_sum);
    }

    if (threadIdx.x == 0) {
        float slice_mean = local_sum * inv_DHW;
        out[n * C + c] = a_coeff[c] * slice_mean + b_coeff[c];
    }
}

torch::Tensor fused_scale_bn_globalavgpool_cuda(
    torch::Tensor input,
    float scale_factor,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    float momentum,
    bool training
) {
    input = input.contiguous();
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int DHW = D * H * W;
    float inv_DHW = 1.0f / (float)DHW;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto out = torch::empty({N, C, 1, 1, 1}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (training) {
        auto nc_sum_t = torch::empty({N, C}, opts);
        auto sum_t = torch::empty({C}, opts);
        auto sumsq_t = torch::empty({C}, opts);

        int block_size = (DHW > 4096) ? 512 : 256;
        compute_stats_and_nc_sum_kernel<<<C, block_size, 0, stream>>>(
            input.data_ptr<float>(),
            scale_factor,
            nc_sum_t.data_ptr<float>(),
            sum_t.data_ptr<float>(),
            sumsq_t.data_ptr<float>(),
            N, C, DHW
        );

        float n_elements = (float)(N * DHW);
        auto mean_t = sum_t / n_elements;
        auto var_t = sumsq_t / n_elements - mean_t * mean_t;
        auto inv_std_t = torch::rsqrt(var_t + eps);

        running_mean.mul_(1.0f - momentum).add_(mean_t * momentum);
        running_var.mul_(1.0f - momentum).add_(var_t * momentum);

        int nc_total = N * C;
        int block3 = 256;
        int grid3 = (nc_total + block3 - 1) / block3;
        compute_output_from_sums_kernel<<<grid3, block3, 0, stream>>>(
            nc_sum_t.data_ptr<float>(),
            mean_t.data_ptr<float>(),
            inv_std_t.data_ptr<float>(),
            bn_weight.data_ptr<float>(),
            bn_bias.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, inv_DHW
        );
    } else {
        auto running_mean_f32 = running_mean.to(opts.dtype());
        auto running_var_f32 = running_var.to(opts.dtype());
        auto inv_std_t = torch::rsqrt(running_var_f32 + eps);

        auto a_coeff = bn_weight * scale_factor * inv_std_t;
        auto b_coeff = bn_bias - bn_weight * running_mean_f32 * scale_factor * inv_std_t;

        int block_size = 256;
        dim3 grid(C, N);
        eval_affine_reduce_kernel<<<grid, block_size, 0, stream>>>(
            input.data_ptr<float>(),
            a_coeff.data_ptr<float>(),
            b_coeff.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, DHW
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
        and then performs global average pooling. 
        """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        x = self.conv_transpose(x)
        if x.is_cuda and x.dtype == torch.float32:
            x = _stark_get_extension().fused_scale_bn_globalavgpool(
                x,
                float(self.scale_factor),
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                float(self.batch_norm.eps),
                float(self.batch_norm.momentum),
                self.batch_norm.training,
            )
            return x
        else:
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        # scale fused into CUDA kernel above; fallback handled in forward_stmt_1
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # batch norm fused into CUDA kernel above; fallback handled in forward_stmt_1
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # global avg pool fused into CUDA kernel above; fallback handled in forward_stmt_1
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
