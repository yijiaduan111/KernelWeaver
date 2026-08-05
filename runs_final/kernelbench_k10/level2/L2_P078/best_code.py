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
    return f'stark_cuda_l2_p78_{digest}'

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

torch::Tensor fused_pool_pool_sum_cuda(torch::Tensor x);

torch::Tensor fused_pool_pool_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "fused_pool_pool_sum: input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "fused_pool_pool_sum: input must be 5D (N,C,D,H,W)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "fused_pool_pool_sum: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "fused_pool_pool_sum: input must be contiguous");
    return fused_pool_pool_sum_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_pool_pool_sum", &fused_pool_pool_sum, "Fused MaxPool3d-MaxPool3d-Sum CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel: MaxPool3d(k=2,s=2) -> MaxPool3d(k=3,s=3) -> sum over channels
// Input:  [N, C, D,  H,  W]  contiguous float32
// Output: [N, 1, D2, H2, W2] where D2=D1/3, H2=H1/3, W2=W1/3 and D1=D/2 etc.
// Each thread handles one (n, d2, h2, w2) output, loops over C channels.
__global__ void fused_pool_pool_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C,
    int D,  int H,  int W,
    int D1, int H1, int W1,
    int D2, int H2, int W2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D2 * H2 * W2;
    if (idx >= total) return;

    // Decompose linear index
    int w2 = idx % W2; int tmp = idx / W2;
    int h2 = tmp % H2;     tmp = tmp / H2;
    int d2 = tmp % D2;     tmp = tmp / D2;
    int n  = tmp;

    // Base offset in pool1 output space (each pool2 cell covers 3 pool1 cells)
    int d1_base = d2 * 3;
    int h1_base = h2 * 3;
    int w1_base = w2 * 3;

    float channel_sum = 0.0f;

    for (int c = 0; c < C; c++) {
        // Base pointer for this (n, c) slice in input
        long nc_offset = ((long)n * C + c) * (long)(D * H * W);

        // Pool2 window: 3x3x3 over pool1 space
        float p2_max = -3.402823466e+38f;

        for (int kd1 = 0; kd1 < 3; kd1++) {
            int d1 = d1_base + kd1;
            if (d1 >= D1) continue;
            // Each pool1 cell covers 2x2x2 in input
            int d_base = d1 * 2;

            for (int kh1 = 0; kh1 < 3; kh1++) {
                int h1 = h1_base + kh1;
                if (h1 >= H1) continue;
                int h_base = h1 * 2;

                for (int kw1 = 0; kw1 < 3; kw1++) {
                    int w1 = w1_base + kw1;
                    if (w1 >= W1) continue;
                    int w_base = w1 * 2;

                    // Pool1 window: 2x2x2 over input
                    float p1_max = -3.402823466e+38f;
                    for (int id = 0; id < 2; id++) {
                        int di = d_base + id;
                        if (di >= D) continue;
                        for (int ih = 0; ih < 2; ih++) {
                            int hi = h_base + ih;
                            if (hi >= H) continue;
                            for (int iw = 0; iw < 2; iw++) {
                                int wi = w_base + iw;
                                if (wi >= W) continue;
                                float val = __ldg(input + nc_offset + (long)(di * H * W + hi * W + wi));
                                if (val > p1_max) p1_max = val;
                            }
                        }
                    }
                    if (p1_max > p2_max) p2_max = p1_max;
                }
            }
        }
        channel_sum += p2_max;
    }

    long out_idx = ((long)n * D2 + d2) * (long)(H2 * W2) + (long)(h2 * W2 + w2);
    output[out_idx] = channel_sum;
}

torch::Tensor fused_pool_pool_sum_cuda(torch::Tensor x) {
    int N = x.size(0), C = x.size(1);
    int D = x.size(2), H = x.size(3), W = x.size(4);

    // MaxPool3d(k=2, s=2): floor division
    int D1 = D / 2, H1 = H / 2, W1 = W / 2;
    // MaxPool3d(k=3, s=3): floor division
    int D2 = D1 / 3, H2 = H1 / 3, W2 = W1 / 3;

    auto output = torch::zeros({N, 1, D2, H2, W2}, x.options());

    int total = N * D2 * H2 * W2;
    if (total == 0) return output;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_pool_pool_sum_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        D1, H1, W1,
        D2, H2, W2
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.is_contiguous() and x.dtype == torch.float32:
            x = _stark_get_extension().fused_pool_pool_sum(x)
            _fused_tail = True
        else:
            _fused_tail = False
            x = self.max_pool1(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not _fused_tail:
            x = self.max_pool2(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not _fused_tail:
            x = torch.sum(x, dim=1, keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
