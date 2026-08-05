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
    return f'stark_cuda_l1_p2_{digest}'

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

torch::Tensor matmul_cublas(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cublas", &matmul_cublas, "cuBLAS SGEMM (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor matmul_cublas(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul_cublas: inputs must be 2D");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "matmul_cublas: inputs must be float32");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "matmul_cublas: inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "matmul_cublas: inputs must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0), "matmul_cublas: inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    float alpha = 1.0f;
    float beta  = 0.0f;

    // cuBLAS is column-major. For row-major A(M,K) * B(K,N) = C(M,N),
    // use the identity: C^T = B^T * A^T
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), N,
                A.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);

    return C;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a single matrix multiplication (C = A * B)
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        A = A.contiguous()
        B = B.contiguous()
        return _stark_get_extension().matmul_cublas(A, B)
        # <<<END_IMPROVE>>>
