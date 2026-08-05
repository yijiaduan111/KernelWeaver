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
    return f'stark_cuda_l3_p4_{digest}'

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

torch::Tensor lenet5_stage1_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor lenet5_stage2_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lenet5_stage1_forward", &lenet5_stage1_forward, "LeNet5 Stage1: conv1+relu+pool1 (CUDA)");
    m.def("lenet5_stage2_forward", &lenet5_stage2_forward, "LeNet5 Stage2: conv2+relu+pool2 (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Stage 1: conv(1->6, 5x5, valid) + ReLU + maxpool(2x2, stride=2)
// Input:  (N, 1, 32, 32)
// Output: (N, 6, 14, 14)
// Each thread computes one pooled output element at (n, oc, ph, pw)
__global__ __launch_bounds__(256, 4)
void lenet5_stage1_kernel(
    const float* __restrict__ input,   // (N, 1, 32, 32)
    const float* __restrict__ weight,  // (6, 1, 5, 5)
    const float* __restrict__ bias,    // (6,)
    float* __restrict__ output,        // (N, 6, 14, 14)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 6 * 14 * 14;
    if (idx >= total) return;

    int pw = idx % 14;
    int tmp = idx / 14;
    int ph = tmp % 14;
    tmp = tmp / 14;
    int oc = tmp % 6;
    int n  = tmp / 6;

    float b = __ldg(&bias[oc]);

    // Preload the 6x6 input patch
    float patch[6][6];
    int base_r = 2 * ph;
    int base_c = 2 * pw;
    #pragma unroll
    for (int dr = 0; dr < 6; dr++) {
        #pragma unroll
        for (int dc = 0; dc < 6; dc++) {
            patch[dr][dc] = __ldg(&input[n * 1024 + (base_r + dr) * 32 + (base_c + dc)]);
        }
    }

    // Load 5x5 weight for this output channel
    float w[5][5];
    #pragma unroll
    for (int kh = 0; kh < 5; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 5; kw++) {
            w[kh][kw] = __ldg(&weight[oc * 25 + kh * 5 + kw]);
        }
    }

    float pool_max = -1e38f;
    #pragma unroll
    for (int dr2 = 0; dr2 < 2; dr2++) {
        #pragma unroll
        for (int dc2 = 0; dc2 < 2; dc2++) {
            float acc = b;
            #pragma unroll
            for (int kh = 0; kh < 5; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 5; kw++) {
                    acc += patch[dr2 + kh][dc2 + kw] * w[kh][kw];
                }
            }
            acc = acc > 0.f ? acc : 0.f;
            if (acc > pool_max) pool_max = acc;
        }
    }

    output[n * 6 * 196 + oc * 196 + ph * 14 + pw] = pool_max;
}

// Stage 2: conv(6->16, 5x5, valid) + ReLU + maxpool(2x2, stride=2)
// Input:  (N, 6, 14, 14)
// Output: (N, 16, 5, 5)
// Each thread computes one pooled output element at (n, oc, ph, pw).
// Register pressure reduced: stream one input channel at a time through a single
// patch[6][6] buffer plus four persistent accumulators, avoiding patch[6][6][6].
__global__ __launch_bounds__(256, 4)
void lenet5_stage2_kernel(
    const float* __restrict__ input,   // (N, 6, 14, 14)
    const float* __restrict__ weight,  // (16, 6, 5, 5)
    const float* __restrict__ bias,    // (16,)
    float* __restrict__ output,        // (N, 16, 5, 5)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 16 * 5 * 5;
    if (idx >= total) return;

    int pw = idx % 5;
    int tmp = idx / 5;
    int ph = tmp % 5;
    tmp = tmp / 5;
    int oc = tmp % 16;
    int n  = tmp / 16;

    float b = __ldg(&bias[oc]);

    // Four accumulators for the 2x2 pool window conv outputs
    float acc00 = b;
    float acc01 = b;
    float acc10 = b;
    float acc11 = b;

    int base_r = 2 * ph;
    int base_c = 2 * pw;
    int n_input_base = n * 6 * 196;
    int oc_w_base = oc * 150; // oc * 6 * 25

    // Stream one input channel at a time
    #pragma unroll
    for (int ic = 0; ic < 6; ic++) {
        // Load this channel's 6x6 patch into registers
        float patch[6][6];
        int ic_offset = n_input_base + ic * 196 + base_r * 14 + base_c;
        #pragma unroll
        for (int dr = 0; dr < 6; dr++) {
            #pragma unroll
            for (int dc = 0; dc < 6; dc++) {
                patch[dr][dc] = __ldg(&input[ic_offset + dr * 14 + dc]);
            }
        }

        // Load this (oc, ic) weight slice
        float w[5][5];
        int w_base = oc_w_base + ic * 25;
        #pragma unroll
        for (int kh = 0; kh < 5; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 5; kw++) {
                w[kh][kw] = __ldg(&weight[w_base + kh * 5 + kw]);
            }
        }

        // Accumulate all four pool-window offsets
        #pragma unroll
        for (int kh = 0; kh < 5; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 5; kw++) {
                float wv = w[kh][kw];
                acc00 += patch[kh][kw]     * wv;
                acc01 += patch[kh][kw+1]   * wv;
                acc10 += patch[kh+1][kw]   * wv;
                acc11 += patch[kh+1][kw+1] * wv;
            }
        }
    }

    // ReLU then max-pool
    acc00 = acc00 > 0.f ? acc00 : 0.f;
    acc01 = acc01 > 0.f ? acc01 : 0.f;
    acc10 = acc10 > 0.f ? acc10 : 0.f;
    acc11 = acc11 > 0.f ? acc11 : 0.f;

    float pool_max = acc00;
    if (acc01 > pool_max) pool_max = acc01;
    if (acc10 > pool_max) pool_max = acc10;
    if (acc11 > pool_max) pool_max = acc11;

    output[n * 400 + oc * 25 + ph * 5 + pw] = pool_max;
}

torch::Tensor lenet5_stage1_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int N = x.size(0);
    auto output = torch::empty({N, 6, 14, 14}, x.options());
    int total = N * 6 * 14 * 14;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    lenet5_stage1_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}

torch::Tensor lenet5_stage2_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int N = x.size(0);
    auto output = torch::empty({N, 16, 5, 5}, x.options());
    int total = N * 16 * 5 * 5;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    lenet5_stage2_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                LeNet-5 architecture implementation in PyTorch.

                :param num_classes: The number of output classes.
                """
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the LeNet-5 model.

                :param x: The input tensor, shape (batch_size, 1, 32, 32)
                :return: The output tensor, shape (batch_size, num_classes)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().lenet5_stage1_forward(x.contiguous(), self.conv1.weight.contiguous(), self.conv1.bias.contiguous())
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # pooling fused into lenet5_stage1_forward
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = _stark_get_extension().lenet5_stage2_forward(x.contiguous(), self.conv2.weight.contiguous(), self.conv2.bias.contiguous())
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # pooling fused into lenet5_stage2_forward
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        x = x.view(-1, 16*5*5)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        x = F.relu(self.fc1(x))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        x = F.relu(self.fc2(x))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        x = self.fc3(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_10>>>
        return x
        # <<<END_IMPROVE>>>
