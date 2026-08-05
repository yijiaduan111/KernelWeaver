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
    return f'stark_cuda_l2_p75_{digest}'

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

torch::Tensor gemm_groupnorm_min_biasadd_cuda(
    torch::Tensor x,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups,
    double eps,
    torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_groupnorm_min_biasadd_cuda",
          &gemm_groupnorm_min_biasadd_cuda,
          "Fused GroupNorm + Min + BiasAdd (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 256
#define FUSED_BLOCK_SIZE 128
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_min(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void __launch_bounds__(128, 6)
fused_gn_min_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ min_out,
    const int N,
    const int C,
    const int num_groups,
    const int group_size,
    const float eps)
{
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const float* x_row = x + n * C;

    float local_min = FLT_MAX;

    for (int g_base = tid; g_base < num_groups; g_base += blockDim.x) {
        const int g_start = g_base * group_size;

        float mean = 0.f, m2 = 0.f;
        for (int k = 0; k < group_size; ++k) {
            float v = x_row[g_start + k];
            float delta = v - mean;
            mean += delta / (k + 1);
            float delta2 = v - mean;
            m2 += delta * delta2;
        }
        float var = m2 / group_size;
        float inv_std = rsqrtf(var + eps);

        for (int k = 0; k < group_size; ++k) {
            int c = g_start + k;
            float normed = (x_row[c] - mean) * inv_std;
            float val = normed * gn_weight[c] + gn_bias[c];
            if (val < local_min) local_min = val;
        }
    }

    float warp_min = warp_reduce_min(local_min);

    // 128 threads = 4 warps
    __shared__ float warp_mins[4];
    if (lane_id == 0) {
        warp_mins[warp_id] = warp_min;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_min = (lane_id < 4) ? warp_mins[lane_id] : FLT_MAX;
        block_min = warp_reduce_min(block_min);
        if (lane_id == 0) {
            min_out[n] = block_min;
        }
    }
}

__global__ void broadcast_bias_kernel(
    const float* __restrict__ min_in,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N,
    const int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C;
    if (idx >= total) return;

    const int c = idx / N;
    const int n = idx % N;
    out[idx] = min_in[n] + bias[c];
}

torch::Tensor gemm_groupnorm_min_biasadd_cuda(
    torch::Tensor x,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups,
    double eps,
    torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, C]");

    const int N = x.size(0);
    const int C = x.size(1);
    const int gs = C / (int)num_groups;

    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    auto bias_flat = bias.contiguous().view({-1});
    TORCH_CHECK(bias_flat.size(0) == C, "bias must flatten to C elements");

    auto min_temp = torch::empty({N}, x.options());

    dim3 grid1(N);
    dim3 block1(FUSED_BLOCK_SIZE);
    fused_gn_min_kernel<<<grid1, block1>>>(
        x.data_ptr<float>(),
        gn_weight.contiguous().data_ptr<float>(),
        gn_bias.contiguous().data_ptr<float>(),
        min_temp.data_ptr<float>(),
        N, C, (int)num_groups, gs, (float)eps
    );

    auto out = torch::empty({1, C, N, 1}, x.options());

    const int total_out = N * C;
    const int threads = BLOCK_SIZE;
    const int blocks = (total_out + threads - 1) / threads;
    broadcast_bias_kernel<<<blocks, threads>>>(
        min_temp.data_ptr<float>(),
        bias_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
        """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _use_fused = (
        x.is_cuda and
        x.dtype == torch.float32 and
        x.dim() == 2 and
        self.bias.numel() == x.size(1)
        )
        if _use_fused:
            x = _stark_get_extension().gemm_groupnorm_min_biasadd_cuda(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
            self.bias.contiguous()
            )
        else:
            x = self.group_norm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not _use_fused:
                    x = torch.min(x, dim=1, keepdim=True)[0]
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not _use_fused:
                    x = x + self.bias
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
