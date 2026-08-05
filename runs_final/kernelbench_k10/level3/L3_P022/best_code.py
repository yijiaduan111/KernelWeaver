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
    return f'stark_cuda_l3_p22_{digest}'

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

torch::Tensor fused_head_conv_bn_relu_mean(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double bn_eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_head_conv_bn_relu_mean", &fused_head_conv_bn_relu_mean,
          "Fused conv2+bn2+relu+global_mean for EfficientNetB0 head");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel: for each (batch, out_channel) pair, compute
//   1x1 conv over all 49 spatial positions (HxW=7x7)
//   apply eval-mode BN affine + ReLU
//   accumulate spatial mean
// Grid: (N, C_out_tiles) where each block handles one (n, tile) pair
// Block: 128 threads; each thread handles one or more output channels

#define TILE_C 32
#define THREADS 128
#define HW 49

__global__ void fused_head_kernel(
    const float* __restrict__ input,   // [N, C_in, H, W]
    const float* __restrict__ weight,  // [C_out, C_in, 1, 1] -> [C_out, C_in]
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    float* __restrict__ output,        // [N, C_out]
    int N, int C_in, int C_out,
    float eps
) {
    // Each block: one (batch_idx, out_channel_base) pair
    int n = blockIdx.x;
    int c_base = blockIdx.y * TILE_C;
    int tid = threadIdx.x;

    // Shared memory: accumulate spatial sums for TILE_C channels
    __shared__ float ssum[TILE_C];

    // Each thread handles channels c_base + (tid % TILE_C) with stride TILE_C
    // We use THREADS threads to process TILE_C channels x HW positions
    // tid maps to: channel = tid % TILE_C, hw_start = tid / TILE_C
    int c_local = tid % TILE_C;
    int hw_stride = THREADS / TILE_C;  // = 4
    int hw_start = tid / TILE_C;

    int c_out = c_base + c_local;

    float acc = 0.0f;

    if (c_out < C_out) {
        const float* w_row = weight + c_out * C_in;  // [C_in]

        // Precompute BN affine scale and shift
        float inv_std = rsqrtf(bn_var[c_out] + eps);
        float scale = bn_weight[c_out] * inv_std;
        float shift = bn_bias[c_out] - bn_mean[c_out] * scale;

        // input layout: [N, C_in, H, W] with H=W=7
        const float* inp_n = input + n * C_in * HW;

        for (int hw = hw_start; hw < HW; hw += hw_stride) {
            // Compute 1x1 conv: dot product over C_in
            float val = 0.0f;
            // Manual unrolling hint: C_in=320 for this layer
            for (int ci = 0; ci < C_in; ci++) {
                val += inp_n[ci * HW + hw] * w_row[ci];
            }
            // Apply BN affine + ReLU
            val = val * scale + shift;
            val = val > 0.0f ? val : 0.0f;
            acc += val;
        }
    }

    // Now reduce across hw_stride threads that share the same c_local
    // Use warp shuffle to sum across hw_stride=4 threads per channel
    // Threads with same c_local differ by multiples of TILE_C in tid
    // hw_stride=4, so tid=c_local, c_local+32, c_local+64, c_local+96
    acc += __shfl_down_sync(0xffffffff, acc, TILE_C);
    acc += __shfl_down_sync(0xffffffff, acc, TILE_C * 2);

    // Now tid < TILE_C holds the full sum for its channel
    if (hw_start == 0 && c_out < C_out) {
        output[n * C_out + c_out] = acc / (float)HW;
    }
}

torch::Tensor fused_head_conv_bn_relu_mean(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double bn_eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(conv_weight.is_contiguous(), "conv_weight must be contiguous");

    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = conv_weight.size(0);
    int hw = H * W;

    TORCH_CHECK(hw == HW, "Expected 7x7 spatial map, got ", H, "x", W);

    auto output = torch::empty({N, C_out}, input.options());

    // Grid: N x ceil(C_out/TILE_C)
    int c_tiles = (C_out + TILE_C - 1) / TILE_C;
    dim3 grid(N, c_tiles);
    dim3 block(THREADS);

    // Reshape conv_weight from [C_out, C_in, 1, 1] to [C_out, C_in]
    auto w = conv_weight.view({C_out, C_in});

    fused_head_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        w.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        (float)bn_eps
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                EfficientNetB0 architecture implementation in PyTorch.

                :param num_classes: The number of output classes (default is 1000 for ImageNet).
                """
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.blocks = nn.Sequential(
                    # MBConv1 (32, 16, 1, 1)
                    MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
                    # MBConv6 (16, 24, 2, 6)
                    MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
                    # MBConv6 (24, 24, 1, 6)
                    MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
                    # MBConv6 (24, 40, 2, 6)
                    MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
                    # MBConv6 (40, 40, 1, 6)
                    MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
                    # MBConv6 (40, 80, 2, 6)
                    MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
                    # MBConv6 (80, 80, 1, 6)
                    MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
                    # MBConv6 (80, 112, 1, 6)
                    MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
                    # MBConv6 (112, 112, 1, 6)
                    MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
                    # MBConv6 (112, 192, 2, 6)
                    MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
                    # MBConv6 (192, 192, 1, 6)
                    MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
                    # MBConv6 (192, 192, 1, 6)
                    MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
                    # MBConv6 (192, 320, 1, 6)
                    MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
                )
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the EfficientNetB0 model.

                :param x: The input tensor, shape (batch_size, 3, 224, 224)
                :return: The output tensor, shape (batch_size, num_classes)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = F.relu(self.bn1(self.conv1(x)))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.blocks(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if (not self.training and x.is_cuda and x.shape[2] == 7 and x.shape[3] == 7):
            _ext = _stark_get_extension()
            _w = self.conv2.weight.contiguous()
            _x_c = x.contiguous()
            x = _ext.fused_head_conv_bn_relu_mean(
            _x_c,
            _w,
            self.bn2.running_mean,
            self.bn2.running_var,
            self.bn2.weight,
            self.bn2.bias,
            self.bn2.eps
            )
            # x is already [N, C_out], skip the mean step
            _fused_head_done = True
        else:
            x = F.relu(self.bn2(self.conv2(x)))
            _fused_head_done = False
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if not _fused_head_done:
                    x = x.mean(dim=(2, 3), keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        x = torch.flatten(x, 1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        x = self.fc(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        return x
        # <<<END_IMPROVE>>>
