import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return f'stark_cuda_l3_p2_{digest}'

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

torch::Tensor shallow_wide_mlp_forward(
    torch::Tensor x,
    torch::Tensor w0, torch::Tensor b0,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shallow_wide_mlp_forward", &shallow_wide_mlp_forward, "ShallowWideMLP forward (CUDA)");
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
#include <algorithm>

// Fused bias-add + ReLU (in-place)
__global__ void fused_bias_relu_kernel(float* __restrict__ out,
                                        const float* __restrict__ bias,
                                        int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        int col = i % N;
        float v = out[i] + bias[col];
        out[i] = v > 0.f ? v : 0.f;
    }
}

// Bias-add only (final layer, no ReLU)
__global__ void fused_bias_kernel(float* __restrict__ out,
                                   const float* __restrict__ bias,
                                   int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        out[i] += bias[i % N];
    }
}

// cuBLAS handle - lazily initialised singleton
static cublasHandle_t g_handle = nullptr;

static cublasHandle_t get_cublas_handle() {
    if (g_handle == nullptr) {
        cublasCreate(&g_handle);
    }
    cublasSetStream(g_handle, at::cuda::getDefaultCUDAStream());
    return g_handle;
}

// Run a GEMM with pre-transposed weight: W is [K, N] row-major (already transposed).
// Computes C = W^T * A -> but W is stored as [K,N] so W column-major is [N,K].
// We want C[M,N] = A[M,K] @ W[K,N]  (all row-major).
// In cuBLAS column-major convention:
//   C^T[N,M] = W^T[N,K] @ A^T[K,M]
//   => cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, W_ptr, N, A_ptr, K, C_ptr, N)
// where W_ptr points to [K,N] row-major = [N,K] column-major with leading dim N.
static void run_gemm_pretransposed(
    cublasHandle_t handle,
    int N, int M, int K,
    const float* W_t, // [K, N] row-major (pre-transposed weight)
    const float* A,   // [M, K] row-major (input activations)
    float* C          // [M, N] row-major (output)
) {
    const float alpha = 1.f, beta = 0.f;
    // W_t is [K,N] row-major => column-major [N,K], lda=N
    // A   is [M,K] row-major => column-major [K,M], ldb=K
    // C   is [M,N] row-major => column-major [N,M], ldc=N
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                W_t, N,
                A,   K,
                &beta,
                C,   N);
}

torch::Tensor shallow_wide_mlp_forward(
    torch::Tensor x,
    torch::Tensor w0, torch::Tensor b0,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous());
    TORCH_CHECK(w0.is_contiguous() && w1.is_contiguous() && w2.is_contiguous());

    const int M  = (int)x.size(0);
    const int K0 = (int)x.size(1);    // 16384
    const int N0 = (int)w0.size(1);   // 32768  (w0 is [K0, N0] pre-transposed)
    const int N1 = (int)w1.size(1);   // 32768  (w1 is [N0, N1] pre-transposed)
    const int K2 = (int)w2.size(1);   // 16384  (w2 is [N1, K2] pre-transposed)

    auto opts = x.options();
    torch::Tensor h0  = torch::empty({M, N0}, opts);
    torch::Tensor h1  = torch::empty({M, N1}, opts);
    torch::Tensor out = torch::empty({M, K2}, opts);

    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    run_gemm_pretransposed(handle, N0, M, K0,
        w0.data_ptr<float>(), x.data_ptr<float>(), h0.data_ptr<float>());
    {
        int total = M * N0;
        int threads = 256;
        int blocks = std::min((total + threads - 1) / threads, 65535);
        fused_bias_relu_kernel<<<blocks, threads, 0, stream>>>(
            h0.data_ptr<float>(), b0.data_ptr<float>(), M, N0);
    }

    run_gemm_pretransposed(handle, N1, M, N0,
        w1.data_ptr<float>(), h0.data_ptr<float>(), h1.data_ptr<float>());
    {
        int total = M * N1;
        int threads = 256;
        int blocks = std::min((total + threads - 1) / threads, 65535);
        fused_bias_relu_kernel<<<blocks, threads, 0, stream>>>(
            h1.data_ptr<float>(), b1.data_ptr<float>(), M, N1);
    }

    run_gemm_pretransposed(handle, K2, M, N1,
        w2.data_ptr<float>(), h1.data_ptr<float>(), out.data_ptr<float>());
    {
        int total = M * K2;
        int threads = 256;
        int blocks = std::min((total + threads - 1) / threads, 65535);
        fused_bias_kernel<<<blocks, threads, 0, stream>>>(
            out.data_ptr<float>(), b2.data_ptr<float>(), M, K2);
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)
        # Pre-transpose weights once for CUBLAS_OP_N fast path
        net_layers = list(self.network.children())
        linear_indices = [i for i, l in enumerate(net_layers) if isinstance(l, nn.Linear)]
        if len(linear_indices) == 3:
            l0 = net_layers[linear_indices[0]]
            l1 = net_layers[linear_indices[1]]
            l2 = net_layers[linear_indices[2]]
            self.register_buffer('_stark_w0_t', l0.weight.t().contiguous(), persistent=False)
            self.register_buffer('_stark_w1_t', l1.weight.t().contiguous(), persistent=False)
            self.register_buffer('_stark_w2_t', l2.weight.t().contiguous(), persistent=False)
        else:
            self._stark_w0_t = None
            self._stark_w1_t = None
            self._stark_w2_t = None
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        layers = list(self.network.children())
        use_fast_path = (
        x.is_cuda and
        x.dtype == torch.float32 and
        x.is_contiguous() and
        len(layers) == 5 and
        isinstance(layers[0], nn.Linear) and
        isinstance(layers[1], nn.ReLU) and
        isinstance(layers[2], nn.Linear) and
        isinstance(layers[3], nn.ReLU) and
        isinstance(layers[4], nn.Linear) and
        layers[0].in_features == 16384 and layers[0].out_features == 32768 and
        layers[2].in_features == 32768 and layers[2].out_features == 32768 and
        layers[4].in_features == 32768 and layers[4].out_features == 16384
        )
        if use_fast_path:
            ext = _stark_get_extension()
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if use_fast_path and self._stark_w0_t is not None:
            return ext.shallow_wide_mlp_forward(
            x,
            self._stark_w0_t, layers[0].bias.contiguous(),
            self._stark_w1_t, layers[2].bias.contiguous(),
            self._stark_w2_t, layers[4].bias.contiguous()
            )
        return self.network(x)
        # <<<END_IMPROVE>>>
