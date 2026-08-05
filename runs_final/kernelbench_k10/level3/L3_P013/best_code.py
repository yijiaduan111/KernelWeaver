import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return f'stark_cuda_l3_p13_{digest}'

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

torch::Tensor fused_bn_relu_conv1x1_cuda(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    torch::Tensor conv_weight
);

torch::Tensor fused_bn_relu_conv1x1(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    torch::Tensor conv_weight
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    int C_in = x.size(1);
    TORCH_CHECK(bn_weight.size(0) == C_in, "bn_weight size mismatch");
    TORCH_CHECK(conv_weight.dim() == 4 && conv_weight.size(2) == 1 && conv_weight.size(3) == 1, "conv_weight must be (C_out, C_in, 1, 1)");
    TORCH_CHECK(conv_weight.size(1) == C_in, "conv_weight C_in mismatch");
    return fused_bn_relu_conv1x1_cuda(x, bn_weight, bn_bias, running_mean, running_var, eps, conv_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bn_relu_conv1x1", &fused_bn_relu_conv1x1, "Fused BN+ReLU+1x1 conv (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each thread computes one output element (n, out_c, h, w).
// Fuses: BN normalization -> ReLU -> 1x1 conv accumulation
__global__ void fused_bn_relu_conv1x1_kernel(
    const float* __restrict__ x,         // (N, C_in, H, W)
    const float* __restrict__ bn_weight,  // (C_in,)
    const float* __restrict__ bn_bias,    // (C_in,)
    const float* __restrict__ running_mean, // (C_in,)
    const float* __restrict__ running_var,  // (C_in,)
    float eps,
    const float* __restrict__ conv_w,     // (C_out, C_in)
    float* __restrict__ out,              // (N, C_out, H, W)
    int N, int C_in, int C_out, int H, int W
) {
    int hw = H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * hw;
    if (idx >= total) return;

    int w_idx = idx % W;
    int tmp = idx / W;
    int h_idx = tmp % H;
    tmp = tmp / H;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    // Precompute BN scale/shift per input channel and accumulate conv
    float acc = 0.0f;
    int x_base = n * C_in * hw + h_idx * W + w_idx;
    const float* conv_row = conv_w + oc * C_in;

    for (int ic = 0; ic < C_in; ++ic) {
        float var = running_var[ic];
        float mean = running_mean[ic];
        float scale = bn_weight[ic] * rsqrtf(var + eps);
        float shift = bn_bias[ic] - mean * scale;
        float val = x[x_base + ic * hw];
        float bn_val = val * scale + shift;
        float relu_val = bn_val > 0.0f ? bn_val : 0.0f;
        acc += relu_val * conv_row[ic];
    }

    out[n * C_out * hw + oc * hw + h_idx * W + w_idx] = acc;
}

torch::Tensor fused_bn_relu_conv1x1_cuda(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    torch::Tensor conv_weight
) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = conv_weight.size(0);

    auto out = torch::empty({N, C_out, H, W}, x.options());

    // Make sure conv_weight is (C_out, C_in) contiguous view
    auto conv_w_2d = conv_weight.view({C_out, C_in}).contiguous();

    int total = N * C_out * H * W;
    int block = 256;
    int grid = (total + block - 1) / block;

    fused_bn_relu_conv1x1_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        bn_weight.contiguous().data_ptr<float>(),
        bn_bias.contiguous().data_ptr<float>(),
        running_mean.contiguous().data_ptr<float>(),
        running_var.contiguous().data_ptr<float>(),
        (float)eps,
        conv_w_2d.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, C_out, H, W
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                :param num_input_features: The number of input feature maps
                :param num_output_features: The number of output feature maps
                """
        self.transition = nn.Sequential(
                    nn.BatchNorm2d(num_input_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        bn = self.transition[0]
        relu = self.transition[1]
        conv = self.transition[2]
        pool = self.transition[3]
        y = relu(bn(x))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and not self.training and x.dtype == torch.float32 and x.is_contiguous():
            fused = _stark_get_extension().fused_bn_relu_conv1x1(
            x,
            bn.weight.contiguous(),
            bn.bias.contiguous(),
            bn.running_mean.contiguous(),
            bn.running_var.contiguous(),
            bn.eps,
            conv.weight.contiguous()
            )
            return pool(fused)
        return pool(conv(y))
        # <<<END_IMPROVE>>>
