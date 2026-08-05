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
    return f'stark_cuda_l1_p33_{digest}'

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

torch::Tensor batchnorm2d_infer_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batchnorm2d_infer_cuda", &batchnorm2d_infer_cuda, "BatchNorm2d inference (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BN_BLOCK_W 256

__global__ void batchnorm2d_infer_row_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int N, int C, int H, int W
) {
    int nc = blockIdx.z;
    int n  = nc / C;
    int c  = nc % C;
    int h  = blockIdx.y;
    int w  = blockIdx.x * BN_BLOCK_W + threadIdx.x;

    if (n >= N || h >= H || w >= W) return;

    float mean  = running_mean[c];
    float var   = running_var[c];
    float wt    = weight[c];
    float b     = bias[c];
    float scale = wt * rsqrtf(var + eps);
    float shift = b - mean * scale;

    long long idx = ((long long)n * C + c) * H * W + (long long)h * W + w;
    y[idx] = x[idx] * scale + shift;
}

__global__ void batchnorm2d_infer_row_vec4_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int N, int C, int H, int W4
) {
    int nc = blockIdx.z;
    int n  = nc / C;
    int c  = nc % C;
    int h  = blockIdx.y;
    int w4 = blockIdx.x * BN_BLOCK_W + threadIdx.x;

    if (n >= N || h >= H || w4 >= W4) return;

    float mean  = running_mean[c];
    float var   = running_var[c];
    float wt    = weight[c];
    float b     = bias[c];
    float scale = wt * rsqrtf(var + eps);
    float shift = b - mean * scale;

    long long base = ((long long)n * C + c) * H * (W4 * 4) + (long long)h * (W4 * 4);
    const float4* xv = reinterpret_cast<const float4*>(x + base);
    float4* yv       = reinterpret_cast<float4*>(y + base);

    float4 v = xv[w4];
    v.x = v.x * scale + shift;
    v.y = v.y * scale + shift;
    v.z = v.z * scale + shift;
    v.w = v.w * scale + shift;
    yv[w4] = v;
}

torch::Tensor batchnorm2d_infer_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto y = torch::empty_like(x);

    bool use_vec4 = (W % 4 == 0) &&
                    ((reinterpret_cast<std::uintptr_t>(x.data_ptr<float>()) % 16) == 0) &&
                    ((reinterpret_cast<std::uintptr_t>(y.data_ptr<float>()) % 16) == 0);

    if (use_vec4) {
        int W4 = W / 4;
        dim3 block(BN_BLOCK_W);
        dim3 grid((W4 + BN_BLOCK_W - 1) / BN_BLOCK_W, H, N * C);
        batchnorm2d_infer_row_vec4_kernel<<<grid, block>>>(
            x.data_ptr<float>(), y.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            (float)eps, N, C, H, W4
        );
    } else {
        dim3 block(BN_BLOCK_W);
        dim3 grid((W + BN_BLOCK_W - 1) / BN_BLOCK_W, H, N * C);
        batchnorm2d_infer_row_kernel<<<grid, block>>>(
            x.data_ptr<float>(), y.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            (float)eps, N, C, H, W
        );
    }

    return y;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Batch Normalization.
        """
    def __init__(self, num_features: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the BatchNorm layer.

                Args:
                    num_features (int): Number of features in the input tensor.
                """
        self.bn = nn.BatchNorm2d(num_features=num_features)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        if (
            not self.training and
            x.is_cuda and
            x.dtype == torch.float32 and
            x.dim() == 4 and
            x.is_contiguous() and
            self.bn.weight is not None and
            self.bn.bias is not None and
            self.bn.running_mean is not None and
            self.bn.running_var is not None and
            self.bn.running_mean.is_cuda and
            self.bn.weight.is_contiguous() and
            self.bn.bias.is_contiguous() and
            self.bn.running_mean.is_contiguous() and
            self.bn.running_var.is_contiguous()
        ):
            return _stark_get_extension().batchnorm2d_infer_cuda(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                float(self.bn.eps)
            )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return self.bn(x)
        # <<<END_IMPROVE>>>
