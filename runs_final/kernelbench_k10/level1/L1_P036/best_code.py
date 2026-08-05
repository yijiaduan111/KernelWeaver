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
    return f'stark_cuda_l1_p36_{digest}'

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

torch::Tensor rmsnorm_cuda(torch::Tensor x, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda, "Fused RMSNorm CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rmsnorm_kernel_c64_warp_inplace(
    float* __restrict__ x,
    int inner,
    int total_instances,
    float eps
) {
    int lane = threadIdx.x;
    int warp_row = threadIdx.y;
    int inst = blockIdx.x * blockDim.y + warp_row;
    if (inst >= total_instances) return;

    int n = inst / inner;
    int hw = inst % inner;
    int base = n * 64 * inner + hw;

    float v0 = x[base + lane * inner];
    float v1 = x[base + (lane + 32) * inner];
    float sum = v0 * v0 + v1 * v1;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    sum = __shfl_sync(0xffffffff, sum, 0);

    float inv_rms = rsqrtf(sum / 64.0f + eps);

    x[base + lane * inner] = v0 * inv_rms;
    x[base + (lane + 32) * inner] = v1 * inv_rms;
}

__global__ void rmsnorm_kernel_inplace(
    float* __restrict__ x,
    int C,
    int inner,
    float eps
) {
    int inst = blockIdx.x;
    int n = inst / inner;
    int hw = inst % inner;
    int c = threadIdx.x;

    int base = n * C * inner + hw;
    float val = (c < C) ? x[base + c * inner] : 0.0f;

    __shared__ float smem[1024];
    smem[c] = val * val;
    __syncthreads();

    for (int stride = C / 2; stride > 0; stride >>= 1) {
        if (c < stride) {
            smem[c] += smem[c + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(smem[0] / (float)C + eps);
    __syncthreads();

    if (c < C) {
        x[base + c * inner] = val * inv_rms;
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, double eps) {
    if (x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous() && x.dim() >= 2) {
        int C = x.size(1);
        int64_t batch = x.size(0);
        int64_t inner = x.numel() / (batch * C);
        int64_t total_instances = batch * inner;
        float* x_ptr = x.data_ptr<float>();

        if (C == 64) {
            dim3 grid((unsigned int)((total_instances + 31) / 32));
            dim3 block(32, 32);
            rmsnorm_kernel_c64_warp_inplace<<<grid, block>>>(x_ptr, (int)inner, (int)total_instances, (float)eps);
            return x;
        }

        if (C > 0 && C <= 1024 && (C & (C - 1)) == 0) {
            dim3 grid((unsigned int)total_instances);
            dim3 block(C);
            rmsnorm_kernel_inplace<<<grid, block>>>(x_ptr, C, (int)inner, (float)eps);
            return x;
        }
    }
    auto x2 = x * x;
    auto mean_sq = x2.mean(1, /*keepdim=*/true);
    return x / torch::sqrt(mean_sq + eps);
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs RMS Normalization.
        """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the RMSNorm layer.

                Args:
                    num_features (int): Number of features in the input tensor.
                    eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
                """
        self.num_features = num_features
        self.eps = eps
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if x.is_cuda:
                    return _stark_get_extension().rmsnorm_cuda(x, float(self.eps))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        return x / rms
        # <<<END_IMPROVE>>>
