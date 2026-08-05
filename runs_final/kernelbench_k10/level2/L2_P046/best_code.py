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
    return f'stark_cuda_l2_p46_{digest}'

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

torch::Tensor fused_postconv_epilogue(
    torch::Tensor input,
    float sub1,
    float sub2,
    int pool_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_postconv_epilogue", &fused_postconv_epilogue,
          "Fused subtract-tanh-subtract-avgpool epilogue (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include<cuda_runtime.h>

// Specialized kernel for pool_size == 2.
// __launch_bounds__(128, 4) guides the compiler to target 4 blocks/SM and
// trim per-thread register allocation, improving occupancy over the default.
__launch_bounds__(128, 4)
__global__ void fused_epilogue_pool2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    float sub1, float sub2
) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y;
    int nc    = blockIdx.z;
    if (w_out >= W_out || h_out >= H_out) return;

    int c = nc % C;
    int n = nc / C;

    int h_base= h_out * 2;
    int w_base  = w_out * 2;
    int base_nc = (n * C + c) * (H * W);
    int row0    = base_nc + h_base * W + w_base;
    int row1    = row0 + W;

    float2 r0 = *reinterpret_cast<const float2*>(input + row0);
    float2 r1 = *reinterpret_cast<const float2*>(input + row1);

    // Accumulate in two pairs to keep fewer values alive at once,
    // reducing register pressure versus holding all four transformed values.
    float acc = 0.0f;
    acc += tanhf(r0.x - sub1) - sub2;
    acc += tanhf(r0.y - sub1) - sub2;
    acc += tanhf(r1.x - sub1) - sub2;
    acc += tanhf(r1.y - sub1) - sub2;

    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = acc * 0.25f;
}

// Generic fallback kernel for arbitrary pool sizes
__global__ void fused_epilogue_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int pool_size,
    float sub1, float sub2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp   = idx / W_out;
    int h_out = tmp % H_out;
    tmp       = tmp / H_out;
    int c     = tmp % C;
    int n     = tmp / C;

    float acc  = 0.0f;
    int h_base = h_out * pool_size;
    int w_base = w_out * pool_size;

    for (int ph = 0; ph < pool_size; ph++) {
        int h_in = h_base + ph;
        if (h_in >= H) continue;
        for (int pw = 0; pw < pool_size; pw++) {
            int w_in = w_base + pw;
            if (w_in >= W) continue;
            float val = input[((n * C + c) * H + h_in) * W + w_in];
            val -= sub1;
            val= tanhf(val);
            val -= sub2;
            acc += val;
        }
    }

    output[idx] = acc / (float)(pool_size * pool_size);
}

torch::Tensor fused_postconv_epilogue(
    torch::Tensor input,
    float sub1,
    float sub2,
    int pool_size
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    auto inp = input.contiguous();
    TORCH_CHECK(inp.dtype() == torch::kFloat32, "input must be float32");

    const int N     = inp.size(0);
    const int C     = inp.size(1);
    const int H     = inp.size(2);
    const int W     = inp.size(3);
    const int H_out = H / pool_size;
    const int W_out = W / pool_size;

    auto output = torch::empty({N, C, H_out, W_out}, inp.options());

    if (pool_size == 2) {
        const int threads = 128;
        dim3 block(threads, 1, 1);
        dim3 grid((W_out + threads - 1) / threads, H_out, N * C);
        fused_epilogue_pool2_kernel<<<grid, block>>>(
            inp.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W,
            H_out, W_out,
            sub1, sub2
        );
    } else {
        const int total   = N * C * H_out * W_out;
        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;
        fused_epilogue_generic_kernel<<<blocks, threads>>>(
            inp.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W,
            H_out, W_out,
            pool_size,
            sub1, sub2
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
        """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_postconv_epilogue(x, float(self.subtract1_value), float(self.subtract2_value), int(self.kernel_size_pool))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # tanh fused into fused_postconv_epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # second subtraction fused into fused_postconv_epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # avgpool fused into fused_postconv_epilogue above
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
