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
    return f'stark_cuda_l2_p76_{digest}'

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

torch::Tensor gemm_bias_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);

torch::Tensor gemm_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    return gemm_bias_relu_cuda(
        x.contiguous(), weight.contiguous(), bias.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_bias_relu", &gemm_bias_relu, "Fused GEMM + bias + ReLU (cublasLt)");
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
#include <mutex>

#define CUBLASLT_WORKSPACE_SIZE (32 * 1024 * 1024)

namespace {

struct LtStateCache {
    bool initialized = false;
    int64_t cached_M = -1;
    int64_t cached_K = -1;
    int64_t cached_N = -1;

    cublasLtHandle_t ltHandle = nullptr;
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayout_t Cdesc = nullptr;
    cublasLtMatmulAlgo_t algo;
    bool has_algo = false;
    void* workspace = nullptr;

    std::mutex mtx;

    void reset() {
        if (workspace){ cudaFree(workspace); workspace = nullptr; }
        if (Adesc)         { cublasLtMatrixLayoutDestroy(Adesc); Adesc = nullptr; }
        if (Bdesc)         { cublasLtMatrixLayoutDestroy(Bdesc); Bdesc = nullptr; }
        if (Cdesc)         { cublasLtMatrixLayoutDestroy(Cdesc); Cdesc = nullptr; }
        if (operationDesc) { cublasLtMatmulDescDestroy(operationDesc); operationDesc = nullptr; }
        if (ltHandle)      { cublasLtDestroy(ltHandle); ltHandle = nullptr; }
        initialized = false;
        cached_M = cached_K = cached_N = -1;
        has_algo = false;
    }
};

static LtStateCache g_cache;

} // namespace

torch::Tensor gemm_bias_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // x: [M, K], weight: [N, K] (nn.Linear stores as [out, in]), bias: [N]
    // output: [M, N] = relu(x @ weight^T + bias)
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K, "weight K dim mismatch");
    TORCH_CHECK(bias.size(0) == N, "bias size mismatch");

    auto output = torch::empty({M, N}, x.options());

    const float* bias_ptr = bias.data_ptr<float>();
    float alpha = 1.0f, beta = 0.0f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    std::unique_lock<std::mutex> lock(g_cache.mtx);

    bool shape_changed = (M != g_cache.cached_M || K != g_cache.cached_K || N != g_cache.cached_N);
    bool need_init = !g_cache.initialized || shape_changed;

    if (need_init) {
        if (g_cache.initialized) g_cache.reset();

        bool ok = true;

        if (ok && cublasLtCreate(&g_cache.ltHandle) != CUBLAS_STATUS_SUCCESS) ok = false;

        if (ok && cublasLtMatmulDescCreate(&g_cache.operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) ok = false;

        if (ok) {
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
            if (cublasLtMatmulDescSetAttribute(g_cache.operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS) ok = false;
        }

        if (ok) {
            cublasOperation_t transa = CUBLAS_OP_T;
            cublasOperation_t transb = CUBLAS_OP_N;
            if (cublasLtMatmulDescSetAttribute(g_cache.operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                               &transa, sizeof(transa)) != CUBLAS_STATUS_SUCCESS) ok = false;
            if (ok && cublasLtMatmulDescSetAttribute(g_cache.operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                               &transb, sizeof(transb)) != CUBLAS_STATUS_SUCCESS) ok = false;
        }

        // col-major layout: transa=T => A is (K x M) with ld=K, transb=N => B is (K x N) with ld=K, C is (N x M) with ld=N
        if (ok && cublasLtMatrixLayoutCreate(&g_cache.Adesc, CUDA_R_32F, K, M, K) != CUBLAS_STATUS_SUCCESS) ok = false;
        if (ok && cublasLtMatrixLayoutCreate(&g_cache.Bdesc, CUDA_R_32F, K, N, K) != CUBLAS_STATUS_SUCCESS) ok = false;
        if (ok && cublasLtMatrixLayoutCreate(&g_cache.Cdesc, CUDA_R_32F, N, M, N) != CUBLAS_STATUS_SUCCESS) ok = false;

        if (ok && cudaMalloc(&g_cache.workspace, CUBLASLT_WORKSPACE_SIZE) != cudaSuccess) ok = false;

        if (ok) {
            // Set bias pointer before heuristic query so the algo selection accounts for it
            cublasLtMatmulDescSetAttribute(g_cache.operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
&bias_ptr, sizeof(bias_ptr));

            cublasLtMatmulPreference_t preference = nullptr;
            if (cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS) {
                size_t ws_size = CUBLASLT_WORKSPACE_SIZE;
                cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &ws_size, sizeof(ws_size));
                cublasLtMatmulHeuristicResult_t heurResult = {};
                int returnedResults = 0;
                cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
                    g_cache.ltHandle, g_cache.operationDesc,
                    g_cache.Adesc, g_cache.Bdesc, g_cache.Cdesc, g_cache.Cdesc,
                    preference, 1, &heurResult, &returnedResults);
                if (heur_status == CUBLAS_STATUS_SUCCESS && returnedResults > 0) {
                    g_cache.algo = heurResult.algo;
                    g_cache.has_algo = true;
                }
                cublasLtMatmulPreferenceDestroy(preference);
            }
            g_cache.cached_M = M;
            g_cache.cached_K = K;
            g_cache.cached_N = N;
            g_cache.initialized = true;
        } else {
            // Cache init failed: fall back to uncached one-shot path
            g_cache.reset();
            lock.unlock();

            cublasLtHandle_t ltHandle_fb;
            TORCH_CHECK(cublasLtCreate(&ltHandle_fb) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed (fallback)");
            cublasLtMatmulDesc_t opDesc = nullptr;
            cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr;
            TORCH_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "cublasLtMatmulDescCreate failed (fallback)");
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr));
            cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
            cublasLtMatrixLayoutCreate(&Ad, CUDA_R_32F, K, M, K);
            cublasLtMatrixLayoutCreate(&Bd, CUDA_R_32F, K, N, K);
            cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, N, M, N);
            auto ws_buf = torch::empty({CUBLASLT_WORKSPACE_SIZE}, x.options().dtype(torch::kUInt8));
            TORCH_CHECK(cublasLtMatmul(ltHandle_fb, opDesc, &alpha,
                x.data_ptr<float>(), Ad,
                weight.data_ptr<float>(), Bd,
                &beta,
                output.data_ptr<float>(), Cd,
                output.data_ptr<float>(), Cd,
                nullptr,
                ws_buf.data_ptr(), CUBLASLT_WORKSPACE_SIZE,
                stream) == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (fallback) failed");
            cublasLtMatrixLayoutDestroy(Ad);
            cublasLtMatrixLayoutDestroy(Bd);
            cublasLtMatrixLayoutDestroy(Cd);
            cublasLtMatmulDescDestroy(opDesc);
            cublasLtDestroy(ltHandle_fb);
            return output;
        }
    }

    // Update bias pointer for this call (bias data pointer may change between forward calls)
    cublasLtMatmulDescSetAttribute(g_cache.operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                   &bias_ptr, sizeof(bias_ptr));

    if (g_cache.has_algo) {
        TORCH_CHECK(cublasLtMatmul(
            g_cache.ltHandle, g_cache.operationDesc,
            &alpha,
            x.data_ptr<float>(), g_cache.Adesc,
            weight.data_ptr<float>(), g_cache.Bdesc,
            &beta,
            output.data_ptr<float>(), g_cache.Cdesc,
            output.data_ptr<float>(), g_cache.Cdesc,
            &g_cache.algo,
            g_cache.workspace, CUBLASLT_WORKSPACE_SIZE,
            stream) == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (cached) failed");
    } else {
        TORCH_CHECK(cublasLtMatmul(
            g_cache.ltHandle, g_cache.operationDesc,
            &alpha,
            x.data_ptr<float>(), g_cache.Adesc,
            weight.data_ptr<float>(), g_cache.Bdesc,
            &beta,
            output.data_ptr<float>(), g_cache.Cdesc,
            output.data_ptr<float>(), g_cache.Cdesc,
            nullptr,
            g_cache.workspace, CUBLASLT_WORKSPACE_SIZE,
            stream) == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (cached, no algo) failed");
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
        """
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        if (x.is_cuda and x.dtype == torch.float32
                        and self.gemm.weight.dtype == torch.float32
                        and self.bias.dtype == torch.float32
                        and x.dim() == 2
                        and self.gemm.weight.dim() == 2
                        and self.bias.dim() == 1
                        and x.is_contiguous()
                        and self.gemm.weight.is_contiguous()
                        and self.bias.is_contiguous()):
                    try:
                        return _stark_get_extension().gemm_bias_relu(x, self.gemm.weight, self.bias)
                    except Exception:
                        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x + self.bias
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = torch.relu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
