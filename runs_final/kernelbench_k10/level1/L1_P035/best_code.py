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
    return f'stark_cuda_l1_p35_{digest}'

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

torch::Tensor groupnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("groupnorm_forward", &groupnorm_forward_cuda,
          "GroupNorm forward (CUDA Welford)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("num_groups"), py::arg("eps"));
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void welford_merge(
    float &mean_a, float &m2_a, long long &count_a,
    float mean_b, float m2_b, long long count_b)
{
    if (count_b == 0) return;
    long long count_ab = count_a + count_b;
    float delta = mean_b - mean_a;
    mean_a = mean_a + delta * ((float)count_b / (float)count_ab);
    m2_a = m2_a + m2_b + delta * delta * ((float)count_a * (float)count_b / (float)count_ab);
    count_a = count_ab;
}

__global__ void groupnorm_welford_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int G, int HW,
    float eps)
{
    int ng = blockIdx.x;
    int n = ng / G;
    int g = ng % G;
    int CPG = C / G;
    long long elems = (long long)CPG * HW;
    long long base = (long long)n * C * HW + (long long)g * CPG * HW;

    float mean = 0.f, m2 = 0.f;
    long long cnt = 0;
    for (long long i = threadIdx.x; i < elems; i += blockDim.x) {
        int c_local = (int)(i / HW);
        int hw = (int)(i % HW);
        long long idx = base + (long long)c_local * HW + hw;
        float val = __ldg(&input[idx]);
        cnt++;
        float delta = val - mean;
        mean += delta / (float)cnt;
        m2 += delta * (val - mean);
    }

    extern __shared__ float smem[];
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpCount = (blockDim.x + 31) / 32;
    float* smean = smem;
    float* sm2 = smem + warpCount;
    float* scnt = smem + 2 * warpCount;

    for (int offset = 16; offset >= 1; offset >>= 1) {
        float rmean = __shfl_xor_sync(0xffffffff, mean, offset);
        float rm2 = __shfl_xor_sync(0xffffffff, m2, offset);
        float rcnt_f = __shfl_xor_sync(0xffffffff, (float)cnt, offset);
        long long rcnt = (long long)rcnt_f;
        welford_merge(mean, m2, cnt, rmean, rm2, rcnt);
    }

    if (laneId == 0) {
        smean[warpId] = mean;
        sm2[warpId] = m2;
        scnt[warpId] = (float)cnt;
    }
    __syncthreads();

    if (threadIdx.x < warpCount) {
        mean = smean[threadIdx.x];
        m2 = sm2[threadIdx.x];
        cnt = (long long)scnt[threadIdx.x];
    } else {
        mean = 0.f;
        m2 = 0.f;
        cnt = 0;
    }
    if (threadIdx.x < 32) {
        for (int offset = 16; offset >= 1; offset >>= 1) {
            float rmean = __shfl_xor_sync(0xffffffff, mean, offset);
            float rm2 = __shfl_xor_sync(0xffffffff, m2, offset);
            float rcnt_f = __shfl_xor_sync(0xffffffff, (float)cnt, offset);
            long long rcnt = (long long)rcnt_f;
            welford_merge(mean, m2, cnt, rmean, rm2, rcnt);
        }
        if (threadIdx.x == 0) {
            smean[0] = mean;
            sm2[0] = m2;
            scnt[0] = (float)cnt;
        }
    }
    __syncthreads();

    float final_mean = smean[0];
    float final_var = sm2[0] / ((float)scnt[0]);
    float inv_std = rsqrtf(final_var + eps);

    for (long long i = threadIdx.x; i < elems; i += blockDim.x) {
        int c_local = (int)(i / HW);
        int hw = (int)(i % HW);
        int c = g * CPG + c_local;
        long long idx = base + (long long)c_local * HW + hw;
        float val = __ldg(&input[idx]);
        float norm = (val - final_mean) * inv_std;
        float w = __ldg(&weight[c]);
        float b = __ldg(&bias[c]);
        output[idx] = norm * w + b;
    }
}

torch::Tensor groupnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;
    int G = num_groups;
    TORCH_CHECK(C % G == 0, "C must be divisible by num_groups");

    const int BLOCK = 512;
    int grid = N * G;
    int warpCount = (BLOCK + 31) / 32;
    size_t smem_bytes = 3 * warpCount * sizeof(float);

    groupnorm_welford_kernel<<<grid, BLOCK, smem_bytes>>>(
        input.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, G, HW, eps);

    return input;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Group Normalization.
        """
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        self.num_features = num_features
        self.num_groups   = num_groups
        self.eps          = self.gn.eps
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()):
                    return _stark_get_extension().groupnorm_forward(
                        x,
                        self.gn.weight,
                        self.gn.bias,
                        self.num_groups,
                        self.eps,
                    )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return self.gn(x)
        # <<<END_IMPROVE>>>
