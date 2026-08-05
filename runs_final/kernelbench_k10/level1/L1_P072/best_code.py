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
    return f'stark_cuda_l1_p72_{digest}'

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

torch::Tensor stark_conv_transpose3d_fast(torch::Tensor input, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_fast", &stark_conv_transpose3d_fast,
          "Specialized grouped ConvTranspose3d forward (float32 CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define IN_C   32
#define OUT_C  32
#define IN_D   12
#define IN_H   24
#define IN_W   48
#define OUT_D  24
#define OUT_H  48
#define OUT_W  96
#define KD     3
#define KH     5
#define KW     7
#define SD     2
#define SH     2
#define SW     2
#define PD     1
#define PH     2
#define PW     3
#define GROUPS 4
#define CIN_PER_GROUP  8
#define COUT_PER_GROUP 8

// OUT_D=24, PD=1: od in [0,23], od_pad = od+1 in [1,24]
//   od_pad odd  (DEPTH_PHASE=1): od = 0,2,4,...,22 => 12 values
//   od_pad even (DEPTH_PHASE=0): od = 1,3,5,...,23 => 12 values
// OUT_H=48, PH=2: oh in [0,47], oh_pad = oh+2 in [2,49]
//   oh_pad even (HEIGHT_PHASE=0): oh = 0,2,...,46 => 24 values
//   oh_pad odd  (HEIGHT_PHASE=1): oh = 1,3,...,47 => 24 values
// OUT_W=96, PW=3: WIDTH_PHASE=0 => ow even, WIDTH_PHASE=1 => ow odd
//   kw_start: WIDTH_PHASE=0 => (PW&1)=1; WIDTH_PHASE=1 => 0

// od recovery from od_local:
//   DEPTH_PHASE=1 (od_pad odd  => od+1 odd  => od even): od = 2*od_local
//   DEPTH_PHASE=0 (od_pad even => od+1 even => od odd):  od = 2*od_local + 1
// oh recovery from oh_local:
//   HEIGHT_PHASE=0 (oh_pad even => oh+2 even => oh even): oh = 2*oh_local
//   HEIGHT_PHASE=1 (oh_pad odd  => oh+2 odd  => oh odd):  oh = 2*oh_local + 1

#define OD_COUNT_D0 12  // od odd,  DEPTH_PHASE=0
#define OD_COUNT_D1 12  // od even, DEPTH_PHASE=1
#define OH_COUNT_H0 24  // oh even, HEIGHT_PHASE=0
#define OH_COUNT_H1 24  // oh odd,  HEIGHT_PHASE=1

template <int WIDTH_PHASE, int DEPTH_PHASE, int HEIGHT_PHASE>
__global__ void conv_transpose3d_phase_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N)
{
    // ow = 2*col + WIDTH_PHASE
    int col = blockIdx.z * blockDim.x + threadIdx.x;
    int ow  = 2 * col + WIDTH_PHASE;
    if (ow >= OUT_W) return;

    int nc_idx = blockIdx.x;
    int n   = nc_idx / OUT_C;
    int oc  = nc_idx % OUT_C;
    if (n >= N) return;

    // Decode od_local / oh_local from blockIdx.y
    // od_count and oh_count are compile-time constants per parity
    const int od_count = (DEPTH_PHASE == 1) ? OD_COUNT_D1 : OD_COUNT_D0;
    const int oh_count = (HEIGHT_PHASE == 0) ? OH_COUNT_H0 : OH_COUNT_H1;

    int dh_idx  = blockIdx.y * blockDim.y + threadIdx.y;
    int od_local = dh_idx / oh_count;
    int oh_local = dh_idx % oh_count;
    if (od_local >= od_count) return;

    // Recover actual od, oh
    int od = (DEPTH_PHASE == 1)  ? (2 * od_local)     : (2 * od_local + 1);
    int oh = (HEIGHT_PHASE == 0) ? (2 * oh_local)     : (2 * oh_local + 1);

    int g        = oc / COUT_PER_GROUP;
    int oc_local = oc % COUT_PER_GROUP;
    int ic_base  = g * CIN_PER_GROUP;

    int od_pad = od + PD;  // parity = DEPTH_PHASE  (compile-time)
    int oh_pad = oh + PH;  // parity = HEIGHT_PHASE (compile-time)
    int ow_pad = ow + PW;  // parity fixed by WIDTH_PHASE

    // Compile-time kd/kh/kw starts
    const int kd_start = DEPTH_PHASE;
    const int kh_start = HEIGHT_PHASE;
    const int kw_start = (WIDTH_PHASE == 0) ? (PW & 1) : ((PW + 1) & 1);

    const int input_n_base = n * IN_C * IN_D * IN_H * IN_W;
    const int out_idx = ((n * OUT_C + oc) * OUT_D + od) * OUT_H * OUT_W + oh * OUT_W + ow;

    float acc = 0.0f;

    #pragma unroll
    for (int ic = 0; ic < CIN_PER_GROUP; ++ic) {
        int global_ic = ic_base + ic;
        const float* weight_ic_ptr = weight + (global_ic * COUT_PER_GROUP + oc_local) * KD * KH * KW;

        #pragma unroll
        for (int kd = kd_start; kd < KD; kd += SD) {
            int id = (od_pad - kd) >> 1;
            if ((unsigned)id >= (unsigned)IN_D) continue;

            #pragma unroll
            for (int kh = kh_start; kh < KH; kh += SH) {
                int ih = (oh_pad - kh) >> 1;
                if ((unsigned)ih >= (unsigned)IN_H) continue;

                const float* input_row  = input + input_n_base + ((global_ic * IN_D + id) * IN_H + ih) * IN_W;
                const float* weight_row = weight_ic_ptr + (kd * KH + kh) * KW;

                #pragma unroll
                for (int kw = kw_start; kw < KW; kw += SW) {
                    int iw = (ow_pad - kw) >> 1;
                    if ((unsigned)iw >= (unsigned)IN_W) continue;
                    acc += input_row[iw] * weight_row[kw];
                }
            }
        }
    }

    output[out_idx] = acc;
}

torch::Tensor stark_conv_transpose3d_fast(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32,
                "Tensors must be float32");
    int N = input.size(0);
    TORCH_CHECK(input.size(1) == IN_C && input.size(2) == IN_D &&
                input.size(3) == IN_H && input.size(4) == IN_W,
                "Input shape mismatch for specialized kernel");
    TORCH_CHECK(weight.size(0) == IN_C && weight.size(1) == COUT_PER_GROUP &&
                weight.size(2) == KD && weight.size(3) == KH && weight.size(4) == KW,
                "Weight shape mismatch for specialized kernel");

    auto output = torch::zeros({N, OUT_C, OUT_D, OUT_H, OUT_W}, input.options());

    dim3 block(32, 8, 1);

    // Width-phase column counts
    int cols0 = (OUT_W + 1) / 2;  // 48 (even ow)
    int cols1 = OUT_W / 2;        // 48 (odd  ow)

    // od counts per DEPTH_PHASE: both = 12
    // oh counts per HEIGHT_PHASE: both = 24
    const int od0 = OD_COUNT_D0, od1 = OD_COUNT_D1;
    const int oh0 = OH_COUNT_H0, oh1 = OH_COUNT_H1;

    // Launch all 8 parity combinations
    // (WIDTH_PHASE, DEPTH_PHASE, HEIGHT_PHASE)
    // Grid: x = N*OUT_C, y = ceil(od_count*oh_count / block.y), z = ceil(cols / block.x)

#define LAUNCH(WP, DP, HP, OD_CNT, OH_CNT, COLS) \
    { \
        int dh_tiles = ((OD_CNT) * (OH_CNT) + block.y - 1) / block.y; \
        dim3 grid(N * OUT_C, dh_tiles, ((COLS) + block.x - 1) / block.x); \
        conv_transpose3d_phase_kernel<WP, DP, HP><<<grid, block>>>( \
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N); \
    }

    LAUNCH(0, 0, 0, od0, oh0, cols0)
    LAUNCH(0, 0, 1, od0, oh1, cols0)
    LAUNCH(0, 1, 0, od1, oh0, cols0)
    LAUNCH(0, 1, 1, od1, oh1, cols0)
    LAUNCH(1, 0, 0, od0, oh0, cols1)
    LAUNCH(1, 0, 1, od0, oh1, cols1)
    LAUNCH(1, 1, 0, od1, oh0, cols1)
    LAUNCH(1, 1, 1, od1, oh1, cols1)

#undef LAUNCH

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (tuple of ints): Size of the convolution kernel in the form (kernel_size_depth, kernel_size_height, kernel_size_width).
            stride (tuple of ints, optional): Stride of the convolution in the form (stride_depth, stride_height, stride_width). Defaults to (1, 1, 1).
            padding (tuple of ints, optional): Padding applied to the input in the form (padding_depth, padding_height, padding_width). Defaults to (0, 0, 0).
            output_padding (tuple of ints, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias
        )
        self._stark_in_channels = in_channels
        self._stark_out_channels = out_channels
        self._stark_kernel_size = tuple(kernel_size)
        self._stark_stride = tuple(stride)
        self._stark_padding = tuple(padding)
        self._stark_output_padding = tuple(output_padding)
        self._stark_groups = groups
        self._stark_bias_enabled = bias
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the 3D transposed convolution.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (
            x.is_cuda and
            x.dtype == torch.float32 and
            x.is_contiguous() and
            self.conv_transpose3d.weight.is_contiguous() and
            not self._stark_bias_enabled and
            self._stark_in_channels == 32 and
            self._stark_out_channels == 32 and
            self._stark_kernel_size == (3, 5, 7) and
            self._stark_stride == (2, 2, 2) and
            self._stark_padding == (1, 2, 3) and
            self._stark_output_padding == (1, 1, 1) and
            self._stark_groups == 4 and
            tuple(x.shape[1:]) == (32, 12, 24, 48)
        ):
            return _stark_get_extension().conv_transpose3d_fast(
                x, self.conv_transpose3d.weight
            )
        return self.conv_transpose3d(x)
        # <<<END_IMPROVE>>>
