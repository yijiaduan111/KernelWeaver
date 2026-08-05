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
    return f'stark_cuda_l2_p81_{digest}'

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

torch::Tensor fused_post_gemm_cuda(torch::Tensor x);

torch::Tensor fused_post_gemm(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "fused_post_gemm: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "fused_post_gemm: input must be float32");
    if (!x.is_contiguous()) x = x.contiguous();
    return fused_post_gemm_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_gemm", &fused_post_gemm, "Fused swish/divide/clamp/tanh/clamp (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float epilogue(float x) {
    x = x * (1.0f / (1.0f + __expf(-x)));
    x *= 0.5f;
    x = fminf(1.0f, fmaxf(-1.0f, x));
    return tanhf(x);  // tanh output is in [-1,1]; post-tanh clamp is redundant
}

__global__ void __launch_bounds__(256, 4)
fused_post_gemm_kernel_vec(float* __restrict__ data, int n4, int n_tail) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n4) {
        float4 v = reinterpret_cast<float4*>(data)[idx];
        v.x = epilogue(v.x);
        v.y = epilogue(v.y);
        v.z = epilogue(v.z);
        v.w = epilogue(v.w);
        reinterpret_cast<float4*>(data)[idx] = v;
    }

    int tail_idx = n4 * 4 + idx;
    if (idx < n_tail) {
        data[tail_idx] = epilogue(data[tail_idx]);
    }
}

torch::Tensor fused_post_gemm_cuda(torch::Tensor x) {
    int total = x.numel();
    int n4 = total / 4;
    int n_tail = total % 4;

    int block = 256;
    int grid = (n4 + block - 1) / block;
    if (grid == 0 && n_tail > 0) grid = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_post_gemm_kernel_vec<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), n4, n_tail
    );

    return x;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
        """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            # Keep the pre-transposed weight as a registered buffer so .to(device) moves it with the module.
            self.register_buffer("weight_t", self.gemm.weight.t().contiguous())
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_features).
                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_features).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if self.gemm.bias is not None:
            x = torch.addmm(self.gemm.bias, x, self.weight_t)
        else:
            x = torch.mm(x, self.weight_t)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if not x.is_contiguous():
            x = x.contiguous()
        x = _stark_get_extension().fused_post_gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # divide by 2 is handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # clamp [-1,1] is handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # tanh is handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        # final clamp [-1,1] is handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        return x
        # <<<END_IMPROVE>>>
