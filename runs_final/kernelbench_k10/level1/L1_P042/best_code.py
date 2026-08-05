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
    return f'stark_cuda_l1_p42_{digest}'

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

torch::Tensor maxpool2d_forward_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool2d_forward_cuda", &maxpool2d_forward_cuda, "Max Pool 2D forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Thread-coarsened kernel with sliding register reuse:
// Each thread computes 4 adjacent output elements along W.
// Fixed params: kernel_size=4, stride=1, padding=1, dilation=1
// For each valid input row, load 7 unique columns once and derive m0..m3.
__global__ void maxpool2d_coarsened_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out
) {
    // Each thread handles 4 consecutive output columns
    int base_ow = ((int)blockIdx.x * blockDim.x + (int)threadIdx.x) * 4;
    int oh = blockIdx.y;
    int nc = blockIdx.z;  // flattened n*C + c

    if (oh >= H_out || base_ow >= W_out) return;

    int n = nc / C;
    int c = nc % C;

    // h_start for this output row (stride=1, padding=1)
    int h_start = oh - 1;  // oh * stride - padding

    // Base pointer for this (n,c) channel
    const float* ch_ptr = input + ((int64_t)n * C + c) * (int64_t)H * W;

    // Four accumulators
    float m0 = -FLT_MAX, m1 = -FLT_MAX, m2 = -FLT_MAX, m3 = -FLT_MAX;

    // The 7 unique input columns needed for 4 outputs with kernel_size=4, stride=1, padding=1:
    // output ow+0 uses cols: base_ow-1, base_ow, base_ow+1, base_ow+2  (ws0+0..3)
    // output ow+1 uses cols: base_ow,   base_ow+1, base_ow+2, base_ow+3
    // output ow+2 uses cols: base_ow+1, base_ow+2, base_ow+3, base_ow+4
    // output ow+3 uses cols: base_ow+2, base_ow+3, base_ow+4, base_ow+5
    // => 7 unique columns: base_ow-1 .. base_ow+5  (indices t=0..6, iw = base_ow-1+t)

    // Precompute validity flags for the 4 output lanes
    bool valid1 = (base_ow + 1 < W_out);
    bool valid2 = (base_ow + 2 < W_out);
    bool valid3 = (base_ow + 3 < W_out);

    #pragma unroll
    for (int kh = 0; kh < 4; kh++) {
        int ih = h_start + kh;
        if (ih < 0 || ih >= H) continue;
        const float* row_ptr = ch_ptr + (int64_t)ih * W;

        // Load 7 unique columns into registers
        float v0, v1, v2, v3, v4, v5, v6;
        int iw;

        iw = base_ow - 1;
        v0 = (iw >= 0 && iw < W) ? __ldg(row_ptr + iw) : -FLT_MAX;
        iw = base_ow;
        v1 = (iw >= 0 && iw < W) ? __ldg(row_ptr + iw) : -FLT_MAX;
        iw = base_ow + 1;
        v2 = (iw >= 0 && iw < W) ? __ldg(row_ptr + iw) : -FLT_MAX;
        iw = base_ow + 2;
        v3 = (iw >= 0 && iw < W) ? __ldg(row_ptr + iw) : -FLT_MAX;
        iw = base_ow + 3;
        v4 = (iw >= 0 && iw < W) ? __ldg(row_ptr + iw) : -FLT_MAX;
        iw = base_ow + 4;
        v5 = (iw >= 0 && iw < W) ? __ldg(row_ptr + iw) : -FLT_MAX;
        iw = base_ow + 5;
        v6 = (iw >= 0 && iw < W) ? __ldg(row_ptr + iw) : -FLT_MAX;

        // Update maxima from cached registers
        // m0: uses v0, v1, v2, v3
        float r0 = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
        if (r0 > m0) m0 = r0;

        // m1: uses v1, v2, v3, v4
        if (valid1) {
            float r1 = fmaxf(fmaxf(v1, v2), fmaxf(v3, v4));
            if (r1 > m1) m1 = r1;
        }

        // m2: uses v2, v3, v4, v5
        if (valid2) {
            float r2 = fmaxf(fmaxf(v2, v3), fmaxf(v4, v5));
            if (r2 > m2) m2 = r2;
        }

        // m3: uses v3, v4, v5, v6
        if (valid3) {
            float r3 = fmaxf(fmaxf(v3, v4), fmaxf(v5, v6));
            if (r3 > m3) m3 = r3;
        }
    }

    // Write outputs
    int64_t out_base = ((int64_t)nc * H_out + oh) * W_out;
    output[out_base + base_ow] = m0;
    if (valid1) output[out_base + base_ow + 1] = m1;
    if (valid2) output[out_base + base_ow + 2] = m2;
    if (valid3) output[out_base + base_ow + 3] = m3;
}

torch::Tensor maxpool2d_forward_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Fixed params: kernel_size=4, stride=1, padding=1, dilation=1
    const int kernel_size = 4;
    const int stride = 1;
    const int padding = 1;
    const int dilation = 1;

    int H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    const int threads = 128;
    // Each thread covers 4 output columns
    int grid_x = (W_out + threads * 4 - 1) / (threads * 4);
    int grid_y = H_out;
    int grid_z = N * C;

    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(threads, 1, 1);

    maxpool2d_coarsened_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Max Pooling 2D.
        """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the Max Pooling 2D layer.

                Args:
                    kernel_size (int): Size of the pooling window.
                    stride (int): Stride of the pooling window.
                    padding (int): Padding to be applied before pooling.
                    dilation (int): Spacing between kernel elements.
                """
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies Max Pooling 2D to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

                Returns:
                    torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        ks = self.maxpool.kernel_size
        st = self.maxpool.stride
        pa = self.maxpool.padding
        di = self.maxpool.dilation
        if (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.dim() == 4
                and ks == 4 and st == 1 and pa == 1 and di == 1):
            return _stark_get_extension().maxpool2d_forward_cuda(x)
        return self.maxpool(x)
        # <<<END_IMPROVE>>>
