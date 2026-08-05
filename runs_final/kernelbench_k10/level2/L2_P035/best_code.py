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
    return f'stark_cuda_l2_p35_{digest}'

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

torch::Tensor fused_epilogue_cuda(torch::Tensor input, float subtract_val, int pool_size, int pool_stride);

torch::Tensor conv_fused_epilogue_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    float subtract_val,
    int pool_size,
    int pool_stride,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dil_h,
    int dil_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_epilogue", &fused_epilogue_cuda, "Fused subtract+HardSwish+MaxPool+Mish (CUDA)");
    m.def("conv_fused_epilogue", &conv_fused_epilogue_cuda, "Conv + Fused epilogue (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float hardswish_fwd(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
}

__device__ __forceinline__ float mish_fwd(float x) {
    float sp = (x > 20.0f) ? x : log1pf(expf(x));
    return x * tanhf(sp);
}

// Generic scalar kernel for arbitrary pool sizes
__global__ void fused_subtract_hardswish_maxpool_mish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const int H_out, const int W_out,
    const int pool_size, const int pool_stride,
    const float subtract_val
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H_out * W_out;
    if (idx >= total) return;

    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int c     = (idx / (W_out * H_out)) % C;
    const int n     = idx / (W_out * H_out * C);

    const int h_start = h_out * pool_stride;
    const int w_start = w_out * pool_stride;
    const int input_nc_offset = (n * C + c) * H * W;

    float max_val = -FLT_MAX;

    for (int ph = 0; ph < pool_size; ph++) {
        const int h_in = h_start + ph;
        if (h_in >= H) continue;
        for (int pw = 0; pw < pool_size; pw++) {
            const int w_in = w_start + pw;
            if (w_in >= W) continue;
            float v = __ldg(&input[input_nc_offset + h_in * W + w_in]) - subtract_val;
            max_val = fmaxf(max_val, hardswish_fwd(v));
        }
    }

    output[(n * C + c) * H_out * W_out + h_out * W_out + w_out] = mish_fwd(max_val);
}

// Specialized kernel for pool_size=2, pool_stride=2 using float2 vectorized row loads
__launch_bounds__(256, 4)
__global__ void fused_sub_hardswish_maxpool2x2_mish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const int H_out, const int W_out,
    const float subtract_val
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H_out * W_out;
    if (idx >= total) return;

    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int c     = (idx / (W_out * H_out)) % C;
    const int n     = idx / (W_out * H_out * C);

    const int h0 = h_out * 2;
    const int w0 = w_out * 2;
    const int nc_off = (n * C + c) * H * W;

    float max_val = -FLT_MAX;

    if (h0 + 1 < H && w0 + 1 < W) {
        const int base0 = nc_off + h0 * W + w0;
        const int base1 = nc_off + (h0 + 1) * W + w0;

        float2 row0 = *reinterpret_cast<const float2*>(input + base0);
        float2 row1 = *reinterpret_cast<const float2*>(input + base1);

        float a = hardswish_fwd(row0.x - subtract_val);
        float b = hardswish_fwd(row0.y - subtract_val);
        float c_val = hardswish_fwd(row1.x - subtract_val);
        float d = hardswish_fwd(row1.y - subtract_val);

        max_val = fmaxf(fmaxf(a, b), fmaxf(c_val, d));
    } else {
        if (h0 < H) {
            const int base0 = nc_off + h0 * W + w0;
            if (w0 + 1 < W) {
                float a = hardswish_fwd(__ldg(input + base0) - subtract_val);
                float b = hardswish_fwd(__ldg(input + base0 + 1) - subtract_val);
                max_val = fmaxf(a, b);
            } else if (w0 < W) {
                max_val = hardswish_fwd(__ldg(input + base0) - subtract_val);
            }
        }
        if (h0 + 1 < H) {
            const int base1 = nc_off + (h0 + 1) * W + w0;
            if (w0 + 1 < W) {
                float a = hardswish_fwd(__ldg(input + base1) - subtract_val);
                float b = hardswish_fwd(__ldg(input + base1 + 1) - subtract_val);
                max_val = fmaxf(max_val, fmaxf(a, b));
            } else if (w0 < W) {
                max_val = fmaxf(max_val, hardswish_fwd(__ldg(input + base1) - subtract_val));
            }
        }
    }

    output[(n * C + c) * H_out * W_out + h_out * W_out + w_out] = mish_fwd(max_val);
}

torch::Tensor fused_epilogue_cuda(
    torch::Tensor input,
    float subtract_val,
    int pool_size,
    int pool_stride
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    input = input.contiguous();

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    const int H_out = (H - pool_size) / pool_stride + 1;
    const int W_out = (W - pool_size) / pool_stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    const int total = N * C * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (pool_size == 2 && pool_stride == 2 && (W & 1) == 0) {
        fused_sub_hardswish_maxpool2x2_mish_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W, H_out, W_out,
            subtract_val
        );
    } else {
        fused_subtract_hardswish_maxpool_mish_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W, H_out, W_out,
            pool_size, pool_stride, subtract_val
        );
    }

    return output;
}

torch::Tensor conv_fused_epilogue_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    float subtract_val,
    int pool_size,
    int pool_stride,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dil_h,
    int dil_w,
    int groups
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");

    auto conv_out = at::conv2d(
        input,
        weight,
        bias,
        {stride_h, stride_w},
        {pad_h, pad_w},
        {dil_h, dil_w},
        groups
    );

    if (!conv_out.is_contiguous()) {
        conv_out = conv_out.contiguous();
    }

    const int N = conv_out.size(0);
    const int C = conv_out.size(1);
    const int H = conv_out.size(2);
    const int W = conv_out.size(3);

    const int H_out = (H - pool_size) / pool_stride + 1;
    const int W_out = (W - pool_size) / pool_stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, conv_out.options());

    const int total = N * C * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (pool_size == 2 && pool_stride == 2 && (W & 1) == 0) {
        fused_sub_hardswish_maxpool2x2_mish_kernel<<<blocks, threads>>>(
            conv_out.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W, H_out, W_out,
            subtract_val
        );
    } else {
        fused_subtract_hardswish_maxpool_mish_kernel<<<blocks, threads>>>(
            conv_out.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W, H_out, W_out,
            pool_size, pool_stride, subtract_val
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
        """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        torch.backends.cudnn.benchmark = True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        ks = self.pool.kernel_size if isinstance(self.pool.kernel_size, int) else self.pool.kernel_size[0]
        st = self.pool.stride
        if st is None:
            st = ks
        elif not isinstance(st, int):
            st = st[0]

        stride = self.conv.stride if isinstance(self.conv.stride, tuple) else (self.conv.stride, self.conv.stride)
        padding = self.conv.padding if isinstance(self.conv.padding, tuple) else (self.conv.padding, self.conv.padding)
        dilation = self.conv.dilation if isinstance(self.conv.dilation, tuple) else (self.conv.dilation, self.conv.dilation)

        ext = _stark_get_extension()
        x = ext.conv_fused_epilogue(
            x,
            self.conv.weight,
            self.conv.bias,
            float(self.subtract_value),
            int(ks),
            int(st),
            int(stride[0]),
            int(stride[1]),
            int(padding[0]),
            int(padding[1]),
            int(dilation[0]),
            int(dilation[1]),
            int(self.conv.groups)
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        # Fused epilogue already applied in forward_stmt_1
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
