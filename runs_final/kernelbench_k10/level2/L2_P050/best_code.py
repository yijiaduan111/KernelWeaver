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
    return f'stark_cuda_l2_p50_{digest}'

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

torch::Tensor fused_epilogue(torch::Tensor conv_out, torch::Tensor scale1, torch::Tensor bias, torch::Tensor scale2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_epilogue", &fused_epilogue, "Fused scale1+avgpool3d+bias+scale2 epilogue (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each thread computes one output voxel [n, c, d_out, h_out, w_out].
// Sequential pointer walk keeps only `p` and `sum` live at any time,
// minimising register pressure vs. materialising all 8 addresses at once.
__global__ void fused_epilogue_kernel(
    const float* __restrict__ inp,   // conv output [N, C, D, H, W]
    const float* __restrict__ bias,  // [C] (bias_shape is [C,1,1,1])
    float* __restrict__ out,         // [N, C, D/2, H/2, W/2]
    float pool_scale,                // scale1 * 0.125f
    float out_scale,                 // scale2
    int N, int C,
    int D, int H, int W,             // input spatial dims
    int D2, int H2, int W2           // output spatial dims = D/2, H/2, W/2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D2 * H2 * W2;
    if (idx >= total) return;

    // Decode linear index -> (n, c, d2, h2, w2)
    int w2 = idx % W2; int tmp = idx / W2;
    int h2 = tmp % H2; tmp /= H2;
    int d2 = tmp % D2; tmp /= D2;
    int c  = tmp % C;
    int n  = tmp / C;

    // Strides for NCDHW layout: inp[n, c, d, h, w]
    int sN = C * D * H * W;
    int sC = D * H * W;
    int sD = H * W;
    // sH == W

    // Base input pointer at (n, c, d2*2, h2*2, w2*2)
    int base = n * sN + c * sC + d2 * 2 * sD + h2 * 2 * W + w2 * 2;

    // Sequential pointer walk: only `p` and running `sum` are live at once.
    const float* p = inp + base;
    float sum = p[0] + p[1];// (d2*2, h2*2, w2*2) and (w2*2+1)
    p += W;// advance to h2*2+1
    sum += p[0] + p[1];               // (d2*2, h2*2+1, w2*2) and (w2*2+1)
    p += (sD - W);                     // advance to d2*2+1, h2*2
    sum += p[0] + p[1];               // (d2*2+1, h2*2, w2*2) and (w2*2+1)
    p += W;                            // advance to h2*2+1
    sum += p[0] + p[1];               // (d2*2+1, h2*2+1, w2*2) and (w2*2+1)

    // Correct formula: ((sum * scale1 / 8) + bias[c]) * scale2
    out[idx] = (sum * pool_scale + bias[c]) * out_scale;
}

torch::Tensor fused_epilogue(torch::Tensor conv_out, torch::Tensor scale1, torch::Tensor bias, torch::Tensor scale2) {
    auto conv_c = conv_out.contiguous();
    TORCH_CHECK(conv_c.is_cuda(), "conv_out must be a CUDA tensor");
    TORCH_CHECK(conv_c.dim() == 5, "conv_out must be 5D");

    int N = conv_c.size(0);
    int C = conv_c.size(1);
    int D = conv_c.size(2);
    int H = conv_c.size(3);
    int W = conv_c.size(4);
    int D2 = D / 2, H2 = H / 2, W2 = W / 2;

    auto output = torch::empty({N, C, D2, H2, W2}, conv_c.options());

    // Precompute scalars on the host to eliminate in-kernel multiplications
    float pool_scale = scale1.item<float>() * 0.125f;
    float out_scale  = scale2.item<float>();

    // Bias may be shaped [C,1,1,1]; flatten to [C] for the kernel
    auto bias_c = bias.contiguous().view({C});

    int total = N * C * D2 * H2 * W2;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_epilogue_kernel<<<blocks, threads>>>(
        conv_c.data_ptr<float>(),
        bias_c.data_ptr<float>(),
        output.data_ptr<float>(),
        pool_scale,
        out_scale,
        N, C, D, H, W, D2, H2, W2
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_epilogue(x.contiguous(), self.scale1, self.bias, self.scale2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # fused into fused_epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # fused into fused_epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # fused into fused_epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
