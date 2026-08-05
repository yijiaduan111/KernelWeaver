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
    return f'stark_cuda_l1_p43_{digest}'

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

torch::Tensor maxpool3d_forward_cuda(torch::Tensor x);

torch::Tensor maxpool3d_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    return maxpool3d_forward_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool3d_forward", &maxpool3d_forward, "Specialized MaxPool3d forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Each thread computes four adjacent output positions along W (ow0..ow3),
// reusing the shared (n,c,od,oh) base and the overlapping kd/kh input rows.
__global__ void maxpool3d_wquad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C,
    int ID, int IH, int IW,
    int OD, int OH, int OW,
    int stride, int padding, int dilation
) {
    int OW_quads = (OW + 3) / 4;
    int total = N * C * OD * OH * OW_quads;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int tmp = idx;
        int wq  = tmp % OW_quads; tmp /= OW_quads;
        int oh  = tmp % OH;       tmp /= OH;
        int od  = tmp % OD;       tmp /= OD;
        int c   = tmp % C;        tmp /= C;
        int n   = tmp;

        int ow0 = wq * 4;
        int ow1 = ow0 + 1;
        int ow2 = ow0 + 2;
        int ow3 = ow0 + 3;

        int id0 = od * stride - padding;
        int ih0 = oh * stride - padding;
        int iw0_base = ow0 * stride - padding;
        // stride=2, so each successive ow adds stride to iw_base
        int iw1_base = iw0_base + stride;
        int iw2_base = iw1_base + stride;
        int iw3_base = iw2_base + stride;

        const float* base = input + (n * C + c) * (ID * IH * IW);

        float mx0 = -FLT_MAX;
        float mx1 = -FLT_MAX;
        float mx2 = -FLT_MAX;
        float mx3 = -FLT_MAX;

        bool valid1 = (ow1 < OW);
        bool valid2 = (ow2 < OW);
        bool valid3 = (ow3 < OW);

        #pragma unroll
        for (int kd = 0; kd < 3; kd++) {
            int id = id0 + kd * dilation;
            if (id < 0 || id >= ID) continue;
            int id_off = id * (IH * IW);
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                int ih = ih0 + kh * dilation;
                if (ih < 0 || ih >= IH) continue;
                int ih_off = ih * IW;
                int row_off = id_off + ih_off;
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    int disp = kw * dilation;
                    int iw0 = iw0_base + disp;
                    int iw1 = iw1_base + disp;
                    int iw2 = iw2_base + disp;
                    int iw3 = iw3_base + disp;
                    if (iw0 >= 0 && iw0 < IW) {
                        float v = __ldg(&base[row_off + iw0]);
                        if (v > mx0) mx0 = v;
                    }
                    if (valid1 && iw1 >= 0 && iw1 < IW) {
                        float v = __ldg(&base[row_off + iw1]);
                        if (v > mx1) mx1 = v;
                    }
                    if (valid2 && iw2 >= 0 && iw2 < IW) {
                        float v = __ldg(&base[row_off + iw2]);
                        if (v > mx2) mx2 = v;
                    }
                    if (valid3 && iw3 >= 0 && iw3 < IW) {
                        float v = __ldg(&base[row_off + iw3]);
                        if (v > mx3) mx3 = v;
                    }
                }
            }
        }

        int out_base = ((n * C + c) * OD + od) * OH * OW + oh * OW;
        output[out_base + ow0] = mx0;
        if (valid1) output[out_base + ow1] = mx1;
        if (valid2) output[out_base + ow2] = mx2;
        if (valid3) output[out_base + ow3] = mx3;
    }
}

torch::Tensor maxpool3d_forward_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 5, "x must be 5D");

    int N  = x.size(0);
    int C  = x.size(1);
    int ID = x.size(2);
    int IH = x.size(3);
    int IW = x.size(4);

    const int kernel_size = 3;
    const int stride      = 2;
    const int padding     = 1;
    const int dilation    = 3;

    int OD = (ID + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int OH = (IH + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int OW = (IW + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({N, C, OD, OH, OW}, x.options());

    int OW_quads = (OW + 3) / 4;
    int total = N * C * OD * OH * OW_quads;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    maxpool3d_wquad_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, ID, IH, IW, OD, OH, OW,
        stride, padding, dilation
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs Max Pooling 3D.
        """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the Max Pooling 3D layer.

                Args:
                    kernel_size (int): Size of the kernel for the max pooling operation.
                    stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
                    padding (int, optional): Padding applied to the input tensor. Defaults to 0.
                    dilation (int, optional): Spacing between kernel elements. Defaults to 1.
                    return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
                    ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
                """
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies Max Pooling 3D to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

                Returns:
                    torch.Tensor: Output tensor with Max Pooling 3D applied.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        def _check_param(attr, val):
            if isinstance(attr, (tuple, list)):
                return all(a == val for a in attr)
            return attr == val

        if (
            x.is_cuda and
            x.dtype == torch.float32 and
            x.is_contiguous() and
            x.dim() == 5 and
            _check_param(self.maxpool.kernel_size, 3) and
            _check_param(self.maxpool.stride, 2) and
            _check_param(self.maxpool.padding, 1) and
            _check_param(self.maxpool.dilation, 3) and
            self.maxpool.return_indices == False and
            self.maxpool.ceil_mode == False
        ):
            return _stark_get_extension().maxpool3d_forward(x)
        return self.maxpool(x)
        # <<<END_IMPROVE>>>
