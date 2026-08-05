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
    return f'stark_cuda_l1_p86_{digest}'

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

torch::Tensor depthwise_conv_cuda(
    torch::Tensor x,
    torch::Tensor dw_weight,
    torch::Tensor dw_bias,
    int stride,
    int padding,
    int dilation
);

torch::Tensor depthwise_conv(
    torch::Tensor x,
    torch::Tensor dw_weight,
    torch::Tensor dw_bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    return depthwise_conv_cuda(x, dw_weight, dw_bias, stride, padding, dilation);
}

torch::Tensor pointwise_conv_cuda(
    torch::Tensor x,
    torch::Tensor pw_weight,
    torch::Tensor pw_bias
);

torch::Tensor pointwise_conv(
    torch::Tensor x,
    torch::Tensor pw_weight,
    torch::Tensor pw_bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    return pointwise_conv_cuda(x, pw_weight, pw_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_conv", &depthwise_conv, "Depthwise conv with shared-memory tiling (CUDA)");
    m.def("pointwise_conv", &pointwise_conv, "Pointwise 1x1 conv with shared-memory weight caching (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------
// Depthwise 3x3 conv with shared-memory halo tiling
// Specialized for stride=1, padding=1, dilation=1
// -----------------------------------------------------------------------

#define DW_TH 16
#define DW_TW 16
#define DW_HALO 1
#define DW_SMEM_H (DW_TH + 2*DW_HALO)
#define DW_SMEM_W (DW_TW + 2*DW_HALO)

__global__ void __launch_bounds__(DW_TH*DW_TW, 4)
depthwise_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dw_weight,
    const float* __restrict__ dw_bias,
    float* __restrict__ output,
    int N, int Cin, int H, int W,
    int tiles_w, int tiles_h,
    bool has_bias
) {
    int idx = blockIdx.x;
    int tw_idx = idx % tiles_w; idx /= tiles_w;
    int th_idx = idx % tiles_h; idx /= tiles_h;
    int c = idx % Cin; idx /= Cin;
    int n = idx;

    if (n >= N) return;

    int out_row_start = th_idx * DW_TH;
    int out_col_start = tw_idx * DW_TW;

    int t = threadIdx.x;
    int tr = t / DW_TW;
    int tc = t % DW_TW;

    __shared__ float smem[DW_SMEM_H * DW_SMEM_W];

    const float* in_ptr = input + (n * Cin + c) * H * W;
    int total_smem = DW_SMEM_H * DW_SMEM_W;
    int threads = DW_TH * DW_TW;

    for (int i = t; i < total_smem; i += threads) {
        int sr = i / DW_SMEM_W;
        int sc = i % DW_SMEM_W;
        int in_r = out_row_start - DW_HALO + sr;
        int in_c = out_col_start - DW_HALO + sc;
        if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
            smem[i] = in_ptr[in_r * W + in_c];
        } else {
            smem[i] = 0.0f;
        }
    }
    __syncthreads();

    int out_r = out_row_start + tr;
    int out_c = out_col_start + tc;

    if (out_r < H && out_c < W) {
        const float* kw = dw_weight + c * 9;
        float val = 0.0f;
        #pragma unroll
        for (int kr = 0; kr < 3; kr++) {
            #pragma unroll
            for (int kc = 0; kc < 3; kc++) {
                val += kw[kr * 3 + kc] * smem[(tr + kr) * DW_SMEM_W + (tc + kc)];
            }
        }
        if (has_bias) val += dw_bias[c];
        output[(n * Cin + c) * H * W + out_r * W + out_c] = val;
    }
}

torch::Tensor depthwise_conv_cuda(
    torch::Tensor x,
    torch::Tensor dw_weight,
    torch::Tensor dw_bias,
    int stride,
    int padding,
    int dilation
) {
    if (stride != 1 || padding != 1 || dilation != 1 ||
        !x.is_contiguous() || !dw_weight.is_contiguous()) {
        return torch::Tensor();
    }

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    if (dw_weight.size(0) != Cin || dw_weight.size(2) != 3 || dw_weight.size(3) != 3) {
        return torch::Tensor();
    }

    auto output = torch::empty({N, Cin, H, W}, x.options());
    bool has_bias = dw_bias.defined() && dw_bias.numel() > 0;

    int tiles_h = (H + DW_TH - 1) / DW_TH;
    int tiles_w = (W + DW_TW - 1) / DW_TW;

    dim3 grid(N * Cin * tiles_h * tiles_w);
    dim3 block(DW_TH * DW_TW);

    depthwise_conv_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        dw_weight.data_ptr<float>(),
        has_bias ? dw_bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, Cin, H, W,
        tiles_w, tiles_h,
        has_bias
    );

    return output;
}

// -----------------------------------------------------------------------
// Pointwise 1x1 conv: cache [Cout, Cin] weights in shared memory,
// each block handles SP_TILE spatial positions and all Cout output channels.
// Block dim: (Cin=64, SP_TILE) so each thread owns one input channel.
// Cin and Cout are template parameters for the fast path.
// -----------------------------------------------------------------------

#define PW_CIN  64
#define PW_COUT 128
#define PW_SP   4   // spatial positions per block

// Each block: blockDim = (PW_CIN, PW_SP)
// grid: (N * H * W / PW_SP, ceil_div)
__global__ void __launch_bounds__(PW_CIN * PW_SP, 8)
pointwise_conv_kernel(
    const float* __restrict__ input,   // [N, Cin, H, W] contiguous NCHW
    const float* __restrict__ pw_weight, // [Cout, Cin] (from [Cout, Cin, 1, 1])
    const float* __restrict__ pw_bias,   // [Cout] or null
    float* __restrict__ output,          // [N, Cout, H, W]
    int N, int H, int W,
    int HW, int NHW,
    bool has_bias
) {
    // Each block handles PW_SP consecutive spatial positions in the flattened NHW dim
    int sp_block = blockIdx.x;  // which group of PW_SP spatial positions
    int sp_base = sp_block * PW_SP;

    int cin_tid  = threadIdx.x;  // 0..PW_CIN-1
    int sp_tid   = threadIdx.y;  // 0..PW_SP-1

    int sp_global = sp_base + sp_tid;  // global spatial index in NHW
    if (sp_global >= NHW) return;

    int n_idx   = sp_global / HW;
    int hw_idx  = sp_global % HW;
    int h_idx   = hw_idx / W;
    int w_idx   = hw_idx % W;

    // Load weights [Cout, Cin] into shared memory
    __shared__ float smem_w[PW_COUT * PW_CIN];

    // Each (cin_tid, sp_tid) thread loads some weight elements
    // Total elements: PW_COUT * PW_CIN = 128*64 = 8192
    // Total threads: PW_CIN * PW_SP = 64*4 = 256
    // Each thread loads 8192/256 = 32 elements
    int tid_flat = sp_tid * PW_CIN + cin_tid;
    int total_w  = PW_COUT * PW_CIN;
    #pragma unroll
    for (int i = tid_flat; i < total_w; i += PW_CIN * PW_SP) {
        smem_w[i] = pw_weight[i];
    }
    __syncthreads();

    // Each thread (cin_tid, sp_tid) reads its input channel value for its spatial position
    // input[n_idx, cin_tid, h_idx, w_idx] = input[n_idx*Cin*HW + cin_tid*HW + hw_idx]
    float in_val = input[n_idx * PW_CIN * HW + cin_tid * HW + hw_idx];

    // Accumulate over Cout output channels - each thread accumulates into register array
    // Then we need to reduce over cin_tid for each cout
    // Strategy: use shared memory for partial sums
    // Each sp_tid owns PW_COUT outputs, each cin_tid contributes to all PW_COUT outputs
    // Use warp-level reduction per output channel

    // Since PW_CIN=64 = 2 warps, we can use __shfl_down_sync for reduction
    // Each thread computes partial dot products for all Cout channels
    // This is memory-intensive in registers; instead use smem for partials

    __shared__ float smem_partial[PW_SP][PW_COUT]; // partial sums per sp_tid, cout

    // Initialize partial sums
    if (cin_tid < PW_COUT) {
        smem_partial[sp_tid][cin_tid] = 0.0f;
    }
    // Handle remaining if PW_COUT > PW_CIN
    #if PW_COUT > PW_CIN
    if (cin_tid + PW_CIN < PW_COUT) {
        smem_partial[sp_tid][cin_tid + PW_CIN] = 0.0f;
    }
    #endif
    __syncthreads();

    // Each cin_tid contributes in_val * w[cout, cin_tid] for all cout
    // Accumulate atomically into smem_partial - but atomics are slow
    // Better: unroll over cout, each thread writes PW_COUT partial products
    // Since PW_CIN=64 threads each add to PW_COUT=128 entries, use atomicAdd on smem
    #pragma unroll 8
    for (int co = 0; co < PW_COUT; co++) {
        atomicAdd(&smem_partial[sp_tid][co], smem_w[co * PW_CIN + cin_tid] * in_val);
    }
    __syncthreads();

    // Write output: each thread with cin_tid < PW_COUT writes its output channel
    if (cin_tid < PW_COUT) {
        float out_val = smem_partial[sp_tid][cin_tid];
        if (has_bias) out_val += pw_bias[cin_tid];
        // output[n_idx, cin_tid, h_idx, w_idx]
        output[n_idx * PW_COUT * HW + cin_tid * HW + hw_idx] = out_val;
    }
    // Handle cin_tid in [64, 127] for Cout=128 > Cin=64
    if (cin_tid + PW_CIN < PW_COUT) {
        int co2 = cin_tid + PW_CIN;
        float out_val = smem_partial[sp_tid][co2];
        if (has_bias) out_val += pw_bias[co2];
        output[n_idx * PW_COUT * HW + co2 * HW + hw_idx] = out_val;
    }
}

torch::Tensor pointwise_conv_cuda(
    torch::Tensor x,
    torch::Tensor pw_weight,
    torch::Tensor pw_bias
) {
    // Fast path: contiguous float32 NCHW, Cin==64, Cout==128, 1x1
    if (!x.is_contiguous() || !pw_weight.is_contiguous()) {
        return torch::Tensor();
    }
    if (x.dim() != 4 || pw_weight.dim() != 4) {
        return torch::Tensor();
    }

    int N    = x.size(0);
    int Cin  = x.size(1);
    int H    = x.size(2);
    int W    = x.size(3);
    int Cout = pw_weight.size(0);

    if (Cin != PW_CIN || Cout != PW_COUT) {
        return torch::Tensor();
    }
    if (pw_weight.size(2) != 1 || pw_weight.size(3) != 1) {
        return torch::Tensor();
    }

    auto output  = torch::empty({N, Cout, H, W}, x.options());
    bool has_bias = pw_bias.defined() && pw_bias.numel() > 0;

    int HW  = H * W;
    int NHW = N * HW;

    int sp_blocks = (NHW + PW_SP - 1) / PW_SP;
    dim3 grid(sp_blocks);
    dim3 block(PW_CIN, PW_SP);

    pointwise_conv_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        pw_weight.data_ptr<float>(),
        has_bias ? pw_bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, H, W, HW, NHW,
        has_bias
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a depthwise-separable 2D convolution operation.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolution kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to the input. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        fast_out = None
        try:
            ext = _stark_get_extension()
            if (
                x.is_cuda
                and x.dtype == torch.float32
                and x.is_contiguous()
                and self.depthwise.groups == self.depthwise.in_channels
                and self.depthwise.out_channels == self.depthwise.in_channels
                and self.depthwise.kernel_size == (3, 3)
                and self.depthwise.stride == (1, 1)
                and self.depthwise.padding == (1, 1)
                and self.depthwise.dilation == (1, 1)
            ):
                dw_bias_t = self.depthwise.bias.contiguous() if self.depthwise.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
                result = ext.depthwise_conv(
                    x,
                    self.depthwise.weight.contiguous(),
                    dw_bias_t,
                    1,
                    1,
                    1,
                )
                if result.numel() > 0:
                    fast_out = result
        except Exception:
            fast_out = None
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if fast_out is not None:
            x = fast_out
        else:
            x = self.depthwise(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.pointwise(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
