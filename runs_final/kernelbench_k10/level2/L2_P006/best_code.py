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
    return f'stark_cuda_l2_p6_{digest}'

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

torch::Tensor softmax_c16_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_c16_cuda", &softmax_c16_cuda,
          "Warp-shuffle numerically-stable softmax over dim=1 for C=16 NCDHW float32 tensors");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Numerically-stable channel softmax for C=16, contiguous NCDHW float32 input.
// Vectorized float4 IO variant:
//   Each warp handles 2 spatial voxels (half-warp per voxel).
//   Each half-warp has 16 active lanes (lanes 0-15 or 16-31).
//   For a given voxel, the 16 channel values are contiguous in memory at stride DHW.
//   We load them as scalars via __ldg (channels are NOT contiguous in NCDHW for a fixed spatial
//   position -- channel c is at offset c*DHW from the base), so float4 vectorization applies
//   to loading 4 consecutive channels at once using gathered float4 from a transposed perspective.
//
// NCDHW layout: element [n,c,d,h,w] = inp[n*C*DHW + c*DHW + spatial]
// For fixed (n, spatial), channels are at stride DHW apart -- NOT contiguous.
// float4 vectorization only helps if channels are contiguous. They are NOT in NCDHW.
// Instead, we use a warp-level approach with one warp per voxel (32 threads, 16 active per half)
// and apply __ldg scalar loads, but improve throughput by packing 2 voxels per warp as before.
//
// Optimization: use one FULL warp per voxel (32 threads, all active, lanes 0-15 handle
// channels 0-15, lanes 16-31 also handle channels 0-15 but a DIFFERENT spatial approach).
// Actually the best vectorization for NCDHW softmax over C=16 is to use a different tile:
// transpose the problem so we load channel-major. We use one warp per 2 voxels (existing),
// but load 4 channels at a time using loop unrolling + __ldg with prefetch hints.
//
// Key insight for float4: if we process the tensor as N*DHW voxels with C=16 channels each,
// a SINGLE voxel's channels are at addresses: base, base+DHW, base+2*DHW, ..., base+15*DHW.
// These are NOT contiguous. float4 requires contiguous addresses.
//
// Better approach: launch with one thread per voxel, load all 16 channels in a loop,
// do sequential softmax in registers. This avoids warp shuffle complexity and uses
// L1 cache efficiently with __ldg prefetching.

__global__ void softmax_c16_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int DHW,
    int NDHW
) {
    // One thread per voxel. Each thread processes C=16 channels.
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_idx >= NDHW) return;

    int n       = voxel_idx / DHW;
    int spatial = voxel_idx - n * DHW;
    // Base pointer for this voxel: inp[n, 0, spatial] = inp[n*16*DHW + spatial]
    const float* base_in  = inp + n * (16 * DHW) + spatial;
    float*       base_out = out + n * (16 * DHW) + spatial;

    // Load all 16 channels into registers
    float v0  = __ldg(base_in +  0 * DHW);
    float v1  = __ldg(base_in +  1 * DHW);
    float v2  = __ldg(base_in +  2 * DHW);
    float v3  = __ldg(base_in +  3 * DHW);
    float v4  = __ldg(base_in +  4 * DHW);
    float v5  = __ldg(base_in +  5 * DHW);
    float v6  = __ldg(base_in +  6 * DHW);
    float v7  = __ldg(base_in +  7 * DHW);
    float v8  = __ldg(base_in +  8 * DHW);
    float v9  = __ldg(base_in +  9 * DHW);
    float v10 = __ldg(base_in + 10 * DHW);
    float v11 = __ldg(base_in + 11 * DHW);
    float v12 = __ldg(base_in + 12 * DHW);
    float v13 = __ldg(base_in + 13 * DHW);
    float v14 = __ldg(base_in + 14 * DHW);
    float v15 = __ldg(base_in + 15 * DHW);

    // Compute max for numerical stability
    float mx = v0;
    mx = fmaxf(mx, v1);  mx = fmaxf(mx, v2);  mx = fmaxf(mx, v3);
    mx = fmaxf(mx, v4);  mx = fmaxf(mx, v5);  mx = fmaxf(mx, v6);
    mx = fmaxf(mx, v7);  mx = fmaxf(mx, v8);  mx = fmaxf(mx, v9);
    mx = fmaxf(mx, v10); mx = fmaxf(mx, v11); mx = fmaxf(mx, v12);
    mx = fmaxf(mx, v13); mx = fmaxf(mx, v14); mx = fmaxf(mx, v15);

    // Compute exp(v - max)
    v0  = __expf(v0  - mx); v1  = __expf(v1  - mx);
    v2  = __expf(v2  - mx); v3  = __expf(v3  - mx);
    v4  = __expf(v4  - mx); v5  = __expf(v5  - mx);
    v6  = __expf(v6  - mx); v7  = __expf(v7  - mx);
    v8  = __expf(v8  - mx); v9  = __expf(v9  - mx);
    v10 = __expf(v10 - mx); v11 = __expf(v11 - mx);
    v12 = __expf(v12 - mx); v13 = __expf(v13 - mx);
    v14 = __expf(v14 - mx); v15 = __expf(v15 - mx);

    // Sum
    float s = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7
            + v8 + v9 + v10 + v11 + v12 + v13 + v14 + v15;
    float inv_s = __frcp_rn(s);

    // Write normalized values
    base_out[ 0 * DHW] = v0  * inv_s;
    base_out[ 1 * DHW] = v1  * inv_s;
    base_out[ 2 * DHW] = v2  * inv_s;
    base_out[ 3 * DHW] = v3  * inv_s;
    base_out[ 4 * DHW] = v4  * inv_s;
    base_out[ 5 * DHW] = v5  * inv_s;
    base_out[ 6 * DHW] = v6  * inv_s;
    base_out[ 7 * DHW] = v7  * inv_s;
    base_out[ 8 * DHW] = v8  * inv_s;
    base_out[ 9 * DHW] = v9  * inv_s;
    base_out[10 * DHW] = v10 * inv_s;
    base_out[11 * DHW] = v11 * inv_s;
    base_out[12 * DHW] = v12 * inv_s;
    base_out[13 * DHW] = v13 * inv_s;
    base_out[14 * DHW] = v14 * inv_s;
    base_out[15 * DHW] = v15 * inv_s;
}

torch::Tensor softmax_c16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(),              "softmax_c16_cuda: input must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "softmax_c16_cuda: input must be float32");
    TORCH_CHECK(x.is_contiguous(),        "softmax_c16_cuda: input must be contiguous");
    TORCH_CHECK(x.dim() == 5,             "softmax_c16_cuda: expected 5-D NCDHW tensor");
    TORCH_CHECK(x.size(1) == 16,          "softmax_c16_cuda: channel count must be 16");

    const int N    = (int)x.size(0);
    const int D    = (int)x.size(2);
    const int H    = (int)x.size(3);
    const int W    = (int)x.size(4);
    const int DHW  = D * H * W;
    const int NDHW = N * DHW;

    auto out = torch::empty_like(x);

    // 256 threads per block, one thread per voxel
    const int threads = 256;
    const int blocks  = (NDHW + threads - 1) / threads;

    softmax_c16_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        DHW, NDHW
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
        """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        if pool_kernel_size == 2:
            self.pool1 = nn.MaxPool3d(kernel_size=4, stride=4)
            self.pool2 = nn.Identity()
        else:
            self.pool1 = nn.MaxPool3d(pool_kernel_size)
            self.pool2 = nn.MaxPool3d(pool_kernel_size)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x: Input tensor of shape (batch_size, in_channels, depth, height, width)
                Returns:
                    Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.shape[1] == 16:
            x = _stark_get_extension().softmax_c16_cuda(x)
        else:
            x = torch.softmax(x, dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = self.pool1(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = self.pool2(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
