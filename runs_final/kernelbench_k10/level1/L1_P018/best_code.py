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
    return f'stark_cuda_l1_p18_{digest}'

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
# Fast path: cuBLASLt layout-steered output for (A.T @ B.T) = (B @ A).T
# The CUDA extension computes B @ A into transposed-stride storage, returning
# the logically transposed result without a separate transpose kernel pass.
# <<<END_IMPROVE>>>

CUDA_CPP_SRC = r"""
# <<<IMPROVE:cuda_cpp>>>
#include <torch/extension.h>

torch::Tensor matmul_with_transposed_both_cuda(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_with_transposed_both_cuda", &matmul_with_transposed_both_cuda,
        "Matmul with both inputs logically transposed via cuBLASLt (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

// Cache for cuBLASLt handle and algorithm (shape-keyed)
struct LtCache {
    cublasLtHandle_t lt_handle = nullptr;
    cublasLtMatmulAlgo_t algo;
    int64_t cached_M = -1, cached_K = -1, cached_Kb = -1;
    bool algo_valid = false;
};

static LtCache g_lt_cache;

torch::Tensor matmul_with_transposed_both_cuda(torch::Tensor A, torch::Tensor B) {
    // A: (M, K), B: (Kb, N)
    // Result: A.T @ B.T = (K, M) @ (N, Kb) -- requires M == N
    // Equivalent: (B @ A).T, B(Kb,N) @ A(M,K) -> (Kb,M), transposed -> (M,Kb)
    // For the benchmark: A=(8192,2048), B=(4096,8192)
    // A.T=(2048,8192), B.T=(8192,4096), result=(2048,4096)
    // B@A=(4096,8192)@(8192,2048)=(4096,2048), .T=(2048,4096) â

    if (!A.is_cuda() || !B.is_cuda() ||
        A.scalar_type() != torch::kFloat32 ||
        B.scalar_type() != torch::kFloat32 ||
        A.dim() != 2 || B.dim() != 2) {
        return at::matmul(A.t(), B.t());
    }

    int64_t M = A.size(0), K = A.size(1);
    int64_t Kb = B.size(0), N = B.size(1);
    // Need N == K for B(Kb,N) @ A(M,K) to work
    if (N != K) {
        return at::matmul(A.t(), B.t());
    }

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    // Allocate output with transposed strides: logical shape (K, Kb) = (M_out, N_out)
    // We want out[i,j] = (B@A).T[i,j] = (B@A)[j,i]
    // Allocate physical storage for (Kb, K) row-major, then view as (K, Kb) col-major
    // Simpler: allocate out with shape (K, Kb) and strides (1, K) -- column-major
    // Then gemm_out = out viewed as (Kb, K) with strides (K, 1) -- row-major
    // gemm_out[i,j] = out[j,i] -- so writing B@A into gemm_out fills out as (B@A).T
    auto out = torch::empty_strided({K, Kb}, {1, K}, Ac.options());
    // gemm_out is a view of the same storage with shape (Kb, K) row-major
    auto gemm_out = out.as_strided({Kb, K}, {K, 1});

    // Initialize cuBLASLt handle once
    if (g_lt_cache.lt_handle == nullptr) {
        cublasLtCreate(&g_lt_cache.lt_handle);
    }
    cublasLtHandle_t lt_handle = g_lt_cache.lt_handle;

    // Compute B @ A into gemm_out using cuBLASLt
    // gemm_out(Kb, K) = B(Kb, N) @ A(M, K), N==K, M==K... wait:
    // B(Kb, N) @ A(M, K): inner dims N and M must match => N == M
    // For benchmark: N=8192, M=8192 â
    if (N != M) {
        return at::matmul(A.t(), B.t());
    }

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t layout_B = nullptr, layout_A = nullptr, layout_C = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    // Create operation descriptor
    cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    // Both inputs are row-major (no transpose needed for B@A in row-major)
    cublasOperation_t op_no_trans = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_no_trans, sizeof(op_no_trans));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_no_trans, sizeof(op_no_trans));

    // Matrix layouts: row-major = col-major transposed
    // For cuBLASLt row-major: use CUBLASLT_ORDER_ROW
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;

    // B layout: (Kb, N) row-major
    cublasLtMatrixLayoutCreate(&layout_B, CUDA_R_32F, Kb, N, N);
    cublasLtMatrixLayoutSetAttribute(layout_B, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));

    // A layout: (M, K) row-major -- this is the second operand
    cublasLtMatrixLayoutCreate(&layout_A, CUDA_R_32F, M, K, K);
    cublasLtMatrixLayoutSetAttribute(layout_A, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));

    // C layout: (Kb, K) row-major
    cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_32F, Kb, K, K);
    cublasLtMatrixLayoutSetAttribute(layout_C, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));

    // Preference and heuristic
    cublasLtMatmulPreferenceCreate(&pref);
    size_t workspace_size = 32 * 1024 * 1024; // 32 MB
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &workspace_size, sizeof(workspace_size));

    // Allocate workspace
    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    cublasLtMatmulHeuristicResult_t heuristic_result = {};
    int returned_results = 0;

    cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc,
                                    layout_B, layout_A, layout_C, layout_C,
                                    pref, 1, &heuristic_result, &returned_results);

    float alpha = 1.0f, beta = 0.0f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (returned_results > 0) {
        cublasLtMatmul(lt_handle, op_desc,
                       &alpha,
                       Bc.data_ptr<float>(), layout_B,
                       Ac.data_ptr<float>(), layout_A,
                       &beta,
                       gemm_out.data_ptr<float>(), layout_C,
                       gemm_out.data_ptr<float>(), layout_C,
                       &heuristic_result.algo,
                       workspace, workspace_size,
                       stream);
    } else {
        // Fallback: use ATen mm
        at::mm_out(gemm_out, Bc, Ac);
    }

    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_C);
    cublasLtMatrixLayoutDestroy(layout_A);
    cublasLtMatrixLayoutDestroy(layout_B);
    cublasLtMatmulDescDestroy(op_desc);

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
        if A.is_cuda and B.is_cuda:
            return _stark_get_extension().matmul_with_transposed_both_cuda(A, B)
        return torch.matmul(A.T, B.T)
        # <<<END_IMPROVE>>>
