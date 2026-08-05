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
    return f'stark_cuda_l2_p13_{digest}'

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

torch::Tensor convtranspose3d_mean_bias_cuda_impl(
    torch::Tensor x,
    torch::Tensor bias);

torch::Tensor convtranspose3d_mean_bias_cuda(
    torch::Tensor x,
    torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    return convtranspose3d_mean_bias_cuda_impl(x, bias);
}

torch::Tensor tanh_scale_cuda_impl(
    torch::Tensor x,
    double scaling_factor);

torch::Tensor tanh_scale_cuda(
    torch::Tensor x,
    double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    return tanh_scale_cuda_impl(x, scaling_factor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convtranspose3d_mean_bias_cuda",
          &convtranspose3d_mean_bias_cuda,
          "Fused depth-mean + bias [N,C,D,H,W] -> [N,C,1,H,W]");
    m.def("tanh_scale_cuda",
          &tanh_scale_cuda,
          "Fused tanh + scale pointwise");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each thread handles one (n,c,h,w) output element.
// Loops over D to accumulate the mean, then adds the per-channel bias.
__global__ void mean_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int D, int H, int W) {

    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w      = idx % W;
    int h      = (idx / W) % H;
    int c      = (idx / (W * H)) % C;
    int n      = idx / (W * H * C);

    int HW     = H * W;
    int DHW    = D * HW;
    int hw_off = h * W + w;

    const float* src = x + (n * C + c) * DHW;
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        sum += src[d * HW + hw_off];
    }
    out[(n * C + c) * HW + hw_off] = sum * (1.0f / (float)D) + bias[c];
}

torch::Tensor convtranspose3d_mean_bias_cuda_impl(
    torch::Tensor x,
    torch::Tensor bias) {

    auto x_c    = x.contiguous();
    auto bias_c = bias.contiguous();

    bool fast_path = (x_c.dim() == 5 &&
                      x_c.size(2) > 0 &&
                      bias_c.numel() == x_c.size(1));

    if (fast_path) {
        int N = (int)x_c.size(0);
        int C = (int)x_c.size(1);
        int D = (int)x_c.size(2);
        int H = (int)x_c.size(3);
        int W = (int)x_c.size(4);

        auto out = torch::empty({N, C, 1, H, W}, x_c.options());

        int total = N * C * H * W;
        int block = 256;
        int grid  = (total + block - 1) / block;

        mean_bias_kernel<<<grid, block>>>(
            x_c.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, D, H, W
        );
        return out;
    }

    // Fallback: ATen ops preserving exact semantics
    return x_c.mean(2, /*keepdim=*/true) + bias_c;
}

// Vectorized float4 kernel: processes 4 elements per thread
__global__ void tanh_scale_kernel_vec4(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    int n4,
    float scale) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n4) return;
    float4 v = x[idx];
    v.x = tanhf(v.x) * scale;
    v.y = tanhf(v.y) * scale;
    v.z = tanhf(v.z) * scale;
    v.w = tanhf(v.w) * scale;
    out[idx] = v;
}

// Scalar fallback kernel for tail elements
__global__ void tanh_scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int total,
    float scale) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = tanhf(x[idx]) * scale;
}

torch::Tensor tanh_scale_cuda_impl(
    torch::Tensor x,
    double scaling_factor) {

    auto x_c = x.contiguous();

    if (x_c.is_cuda() && x_c.dtype() == torch::kFloat32) {
        auto out    = torch::empty_like(x_c);
        int  total  = (int)x_c.numel();
        float scale = (float)scaling_factor;
        int  block  = 256;

        const float* xptr  = x_c.data_ptr<float>();
        float*       optr  = out.data_ptr<float>();

        // Use float4 path when total >= 4 and both pointers are 16-byte aligned
        uintptr_t xaddr = (uintptr_t)xptr;
        uintptr_t oaddr = (uintptr_t)optr;
        int n4    = total / 4;
        int tail  = total % 4;

        if (n4 > 0 && (xaddr & 15) == 0 && (oaddr & 15) == 0) {
            int grid4 = (n4 + block - 1) / block;
            tanh_scale_kernel_vec4<<<grid4, block>>>(
                reinterpret_cast<const float4*>(xptr),
                reinterpret_cast<float4*>(optr),
                n4,
                scale
            );
            if (tail > 0) {
                int grid_tail = 1;
                tanh_scale_kernel<<<grid_tail, block>>>(
                    xptr + n4 * 4,
                    optr  + n4 * 4,
                    tail,
                    scale
                );
            }
        } else {
            int grid = (total + block - 1) / block;
            tanh_scale_kernel<<<grid, block>>>(
                xptr, optr, total, scale
            );
        }
        return out;
    }

    // Fallback
    return torch::tanh(x_c) * scaling_factor;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a series of operations:
        1. Transposed 3D convolution
        2. Mean pooling (across depth)
        3. Addition
        4. Softmax (across channels)
        5. Tanh activation
        6. Scaling
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().convtranspose3d_mean_bias_cuda(x, self.bias)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # Bias add fused into the CUDA mean+bias kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = torch.softmax(x, dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = _stark_get_extension().tanh_scale_cuda(x, self.scaling_factor)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # scaling fused into the CUDA tanh_scale kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
