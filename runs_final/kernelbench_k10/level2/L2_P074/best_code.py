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
    return f'stark_cuda_l2_p74_{digest}'

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

torch::Tensor fused_act_pool(torch::Tensor conv_out, torch::Tensor multiplier);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_act_pool", &fused_act_pool, "Fused LeakyReLU-Multiply-LeakyReLU-MaxPool3d");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__launch_bounds__(128, 4)
__global__ void fused_act_pool_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ multiplier,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    // blockIdx.y encodes the (n, c) slice
    int nc = blockIdx.y;
    int c = nc % C;
    int n = nc / C;

    // Cache multiplier[c] once per block in shared memory
    __shared__ float smem_mult;
    if (threadIdx.x == 0) {
        smem_mult = multiplier[c];
    }
    __syncthreads();

    float mult = smem_mult;

    // Each thread handles one pooled spatial output element
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_total = D_out * H_out * W_out;
    if (spatial_idx >= spatial_total) return;

    int w_out = spatial_idx % W_out;
    int tmp   = spatial_idx / W_out;
    int h_out = tmp % H_out;
    int d_out = tmp / H_out;

    int d_base = d_out * 2;
    int h_base = h_out * 2;
    int w_base = w_out * 2;

    // Precompute base offset for this (n, c) slice to avoid repeated multiply
    int nc_stride = D_in * H_in * W_in;
    int nc_offset = (n * C + c) * nc_stride;

    float max_val = -1e38f;

    for (int kd = 0; kd < 2; kd++) {
        int d = d_base + kd;
        if (d >= D_in) continue;
        for (int kh = 0; kh < 2; kh++) {
            int h = h_base + kh;
            if (h >= H_in) continue;
            for (int kw = 0; kw < 2; kw++) {
                int w = w_base + kw;
                if (w >= W_in) continue;

                int in_idx = nc_offset + d * H_in * W_in + h * W_in + w;
                float v = conv_out[in_idx];

                // LeakyReLU(0.2) -> multiply -> LeakyReLU(0.2)
                v = fmaxf(v, 0.2f * v);
                v = v * mult;
                v = fmaxf(v, 0.2f * v);

                max_val = fmaxf(max_val, v);
            }
        }
    }

    // Output layout: (N, C, D_out, H_out, W_out) => index via nc
    int out_idx = (nc * D_out + d_out) * H_out * W_out + h_out * W_out + w_out;
    output[out_idx] = max_val;
}

torch::Tensor fused_act_pool(torch::Tensor conv_out, torch::Tensor multiplier) {
    TORCH_CHECK(conv_out.is_cuda(), "conv_out must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "multiplier must be a CUDA tensor");
    TORCH_CHECK(conv_out.dtype() == torch::kFloat32, "conv_out must be float32");
    TORCH_CHECK(multiplier.dtype() == torch::kFloat32, "multiplier must be float32");
    TORCH_CHECK(conv_out.dim() == 5, "conv_out must be 5D (NCDHW)");
    TORCH_CHECK(conv_out.is_contiguous(), "conv_out must be contiguous");
    TORCH_CHECK(multiplier.is_contiguous(), "multiplier must be contiguous");

    int N   = conv_out.size(0);
    int C   = conv_out.size(1);
    int D_in = conv_out.size(2);
    int H_in = conv_out.size(3);
    int W_in = conv_out.size(4);

    TORCH_CHECK(multiplier.size(0) == C, "multiplier first dimension must match channels");

    int D_out = D_in / 2;
    int H_out = H_in / 2;
    int W_out = W_in / 2;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, conv_out.options());

    int spatial_total = D_out * H_out * W_out;
    int threads  = 128;
    int blocks_x = (spatial_total + threads - 1) / threads;
    int blocks_y = N * C;

    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads);

    fused_act_pool_kernel<<<grid, block>>>(
        conv_out.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter, 
        applies LeakyReLU again, and performs a max pooling operation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_act_pool(x.contiguous(), self.multiplier.contiguous())
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # fused into fused_act_pool
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # fused into fused_act_pool
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # fused into fused_act_pool
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
