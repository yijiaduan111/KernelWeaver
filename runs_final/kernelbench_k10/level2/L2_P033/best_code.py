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
    return f'stark_cuda_l2_p33_{digest}'

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

torch::Tensor folded_affine_forward(
    torch::Tensor x,
    torch::Tensor eff_scale,
    torch::Tensor eff_bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("folded_affine_forward", &folded_affine_forward,
          "Folded affine transform for eval mode (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 2D shared-memory tiled kernel:
//   blockIdx.x -> channel tile (BLOCK_C channels per block)
//   blockIdx.y -> row tile (each block processes ROWS_PER_BLOCK rows)
// Each block loads eff_scale/eff_bias for its channel tile into smem once,
// then applies them across all assigned rows.

#define BLOCK_C 256
#define ROWS_PER_BLOCK 8

__global__ void folded_affine_smem_kernel(
    const float* __restrict__ x,
    const float* __restrict__ eff_scale,
    const float* __restrict__ eff_bias,
    float* __restrict__ out,
    int N,
    int C
) {
    __shared__ float s_scale[BLOCK_C];
    __shared__ float s_bias[BLOCK_C];

    int c_start = blockIdx.x * BLOCK_C;
    int c = c_start + threadIdx.x;

    // Coalesced load of channel coefficients into shared memory
    if (c < C) {
        s_scale[threadIdx.x] = __ldg(&eff_scale[c]);
        s_bias[threadIdx.x]  = __ldg(&eff_bias[c]);
    } else {
        s_scale[threadIdx.x] = 0.0f;
        s_bias[threadIdx.x]  = 0.0f;
    }
    __syncthreads();

    if (c >= C) return;

    // Row tile: each block handles ROWS_PER_BLOCK consecutive rows
    int row_start = blockIdx.y * ROWS_PER_BLOCK;

    float sc = s_scale[threadIdx.x];
    float bi = s_bias[threadIdx.x];

    // Unrolled over rows
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = row_start + r;
        if (row < N) {
            int idx = row * C + c;
            out[idx] = x[idx] * sc + bi;
        }
    }
}

torch::Tensor folded_affine_forward(
    torch::Tensor x,
    torch::Tensor eff_scale,
    torch::Tensor eff_bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, C]");
    TORCH_CHECK(eff_scale.is_cuda() && eff_scale.is_contiguous(), "eff_scale must be contiguous CUDA");
    TORCH_CHECK(eff_bias.is_cuda() && eff_bias.is_contiguous(), "eff_bias must be contiguous CUDA");
    TORCH_CHECK(eff_scale.scalar_type() == torch::kFloat32, "eff_scale must be float32");
    TORCH_CHECK(eff_bias.scalar_type() == torch::kFloat32, "eff_bias must be float32");

    int N = x.size(0);
    int C = x.size(1);
    TORCH_CHECK(eff_scale.size(0) == C, "eff_scale must have size [C]");
    TORCH_CHECK(eff_bias.size(0) == C, "eff_bias must have size [C]");

    auto out = torch::empty_like(x);

    // grid.x covers channel dimension, grid.y covers row tiles
    dim3 threads(BLOCK_C, 1, 1);
    dim3 blocks(
        (C + BLOCK_C - 1) / BLOCK_C,
        (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
        1
    );

    folded_affine_smem_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        eff_scale.data_ptr<float>(),
        eff_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        C
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
        and then batch normalization.
        """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        _used_folded_eval = False
        if not self.training and x.is_cuda and x.dtype == torch.float32 and x.dim() == 2:
            try:
                eff_scale = (self.scale * self.bn.weight) / torch.sqrt(self.bn.running_var + self.bn.eps)
                eff_bias = self.bn.bias - self.bn.running_mean * eff_scale
                x = _stark_get_extension().folded_affine_forward(
                    x.contiguous(),
                    eff_scale.contiguous(),
                    eff_bias.contiguous()
                )
                _used_folded_eval = True
            except Exception:
                x = x * self.scale
        else:
            x = x * self.scale
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not _used_folded_eval:
            x = self.bn(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        return x
        # <<<END_IMPROVE>>>
