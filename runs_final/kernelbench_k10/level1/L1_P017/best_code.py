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
    return f'stark_cuda_l1_p17_{digest}'

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

torch::Tensor matmul_with_transposed_b_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor matmul_with_transposed_b(torch::Tensor a, torch::Tensor b) {
    return matmul_with_transposed_b_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_with_transposed_b", &matmul_with_transposed_b, "Matmul with logical transposed B (CUDA)");
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

torch::Tensor matmul_with_transposed_b_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    // a: [M, K], b: [N, K], out: [M, N]
    // We compute out = a @ b.T
    // In cuBLAS column-major: C = A^T * B^T where A,B,C are column-major
    // Equivalently: out^T = b @ a^T
    // cuBLAS: C(N,M) = op(B)(N,K) * op(A)(K,M)
    // With op(B)=N (B is [N,K] stored row-major = [K,N] col-major) and op(A)=T
    // We call: C = B * A^T  => C is [N,M] col-major = [M,N] row-major = out

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(0);
    TORCH_CHECK(b.size(1) == K, "b.size(1) must equal a.size(1)");

    auto out = torch::empty({M, N}, a.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS is column-major.
    // We want: out[M,N] = a[M,K] @ b[N,K]^T
    // Treat row-major [M,N] as col-major [N,M].
    // Compute col-major C[N,M] = B[N,K] * A[M,K]^T
    // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // where m=N, n=M, k=K
    // A_cublas = b (row-major [N,K] = col-major [K,N]), transa=N => op(A)=A, lda=K
    // B_cublas = a (row-major [M,K] = col-major [K,M]), transb=T => op(B)=B^T=[M,K]^T=[K,M]->wait
    // Let me redo:
    // C_col[N,M] = B_col[N,K] * A_col[K,M]
    // b row-major [N,K] stored as col-major [K,N]: to get col-major [N,K] we need CUBLAS_OP_T on b
    // a row-major [M,K] stored as col-major [K,M]: to get col-major [K,M] we need CUBLAS_OP_N on a
    // cublasSgemm: C(m,n) = op(A)(m,k) * op(B)(k,n)
    // We want C_col(N,M): m=N, n=M, k=K
    // op(A_cublas)(N,K): A_cublas=b_ptr, CUBLAS_OP_T, lda=N (b is col-major [K,N], lda=K... )
    // This is getting confusing. Use the standard trick:
    // out = a @ b^T  <=>  out^T = b @ a^T
    // In cuBLAS col-major: compute out^T which is [N,M] col-major
    // = b (as [N,K] row-major = [K,N] col-major, need CUBLAS_OP_T to get [N,K])
    //   @ a^T (a is [M,K] row-major = [K,M] col-major, CUBLAS_OP_N gives [K,M], we need [K,M] -> that's a^T in row-major sense)
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
    //             b_ptr, K,   // A in col-major is [K,N], op=T gives [N,K], lda=K
    //             a_ptr, K,   // B in col-major is [K,M], op=N gives [K,M], ldb=K
    //             &beta, out_ptr, N)  // C is [N,M] col-major, ldc=N
    // Result: out_ptr stores [N,M] col-major = [M,N] row-major = desired output

    cublasSgemm(
        handle,
        CUBLAS_OP_T,  // op on b: treat b's col-major [K,N] as [N,K]
        CUBLAS_OP_N,  // op on a: treat a's col-major [K,M] as [K,M]
        (int)N, (int)M, (int)K,
        &alpha,
        b.data_ptr<float>(), (int)K,
        a.data_ptr<float>(), (int)K,
        &beta,
        out.data_ptr<float>(), (int)N
    );

    return out;
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
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs matrix multiplication.

                Args:
                    A: Input tensor of shape (M, K).
                    B: Input tensor of shape (K, N).

                Returns:
                    Output tensor of shape (M, N).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return _stark_get_extension().matmul_with_transposed_b(A, B);
        # <<<END_IMPROVE>>>
