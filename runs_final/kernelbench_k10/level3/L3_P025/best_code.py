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
    return f'stark_cuda_l3_p25_{digest}'

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

torch::Tensor fused_depthwise_bn_shuffle(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_depthwise_bn_shuffle", &fused_depthwise_bn_shuffle,
          "Fused depthwise conv2d + BN affine + channel shuffle (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused depthwise 3x3 conv + BN affine + channel shuffle kernel
// Input:  [N, C, H, W] NCHW, depthwise (groups=C)
// Output: [N, C, H, W] with channels shuffled by `groups`
__global__ void fused_depthwise_bn_shuffle_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,   // [C, 1, 3, 3]
    const float* __restrict__ scale,    // [C]  = bn_weight / sqrt(var + eps)
    const float* __restrict__ bias_out, // [C]  = bn_bias - mean * scale
    float* __restrict__ output,
    int N, int C, int H, int W,
    int groups
) {
    // Each block handles one (n, c, h, w) or we do a flat 1D grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    // Depthwise 3x3 convolution with padding=1
    float val = 0.0f;
    const float* w_ptr = weight + c * 9; // [1,1,3,3] flattened
    for (int kh = 0; kh < 3; kh++) {
        int ih = h + kh - 1;
        for (int kw = 0; kw < 3; kw++) {
            int iw = w + kw - 1;
            float in_val = 0.0f;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                in_val = input[((n * C + c) * H + ih) * W + iw];
            }
            val += w_ptr[kh * 3 + kw] * in_val;
        }
    }

    // Apply BN affine: scale * val + bias
    val = scale[c] * val + bias_out[c];

    // Channel shuffle: c_out = (c % groups) * (C / groups) + (c / groups)
    int C_per_group = C / groups;
    int c_shuffled = (c % groups) * C_per_group + (c / groups);

    output[((n * C + c_shuffled) * H + h) * W + w] = val;
}

torch::Tensor fused_depthwise_bn_shuffle(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps,
    int groups
) {
    input = input.contiguous();
    weight = weight.contiguous();

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Precompute scale and bias on CPU/GPU side
    auto var_eps = (bn_running_var + bn_eps).sqrt();
    auto scale = bn_weight / var_eps;
    auto bias_out = bn_bias - bn_running_mean * scale;

    scale = scale.contiguous();
    bias_out = bias_out.contiguous();

    auto output = torch::empty_like(input);

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_depthwise_bn_shuffle_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias_out.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, groups
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass for ShuffleNet unit.

                :param x: Input tensor, shape (batch_size, in_channels, height, width)
                :return: Output tensor, shape (batch_size, out_channels, height, width)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        out = F.relu(self.bn1(self.conv1(x)))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        fused_shuffle = out.is_cuda and not self.training
        if fused_shuffle:
            out = _stark_get_extension().fused_depthwise_bn_shuffle(
            out,
            self.conv2.weight,
            self.bn2.weight,
            self.bn2.bias,
            self.bn2.running_mean,
            self.bn2.running_var,
            self.bn2.eps,
            self.groups
            )
        else:
            out = self.bn2(self.conv2(out))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not fused_shuffle:
                    out = F.channel_shuffle(out, self.groups)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        out = F.relu(self.bn3(self.conv3(out)))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        out += self.shortcut(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return out
        # <<<END_IMPROVE>>>
