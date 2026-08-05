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
    return f'stark_cuda_l1_p12_{digest}'

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

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diag_matmul_cuda", &diag_matmul_cuda, "Diagonal matrix multiply (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ __launch_bounds__(256, 4) void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * M;
    for (int i = idx; i < total; i += stride) {
        int row = i / M;
        C[i] = __ldg(&A[row]) * __ldg(&B[i]);
    }
}

__global__ __launch_bounds__(256, 4) void diag_matmul_kernel_vec4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * M4;
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);
    for (int i = idx; i < total; i += stride) {
        int row = i / M4;
        float a = __ldg(&A[row]);
        float4 b;
        asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(b.x), "=f"(b.y), "=f"(b.z), "=f"(b.w)
            : "l"(B4 + i));
        float4 c;
        c.x = a * b.x;
        c.y = a * b.y;
        c.z = a * b.z;
        c.w = a * b.w;
        asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
            : : "l"(C4 + i), "f"(c.x), "f"(c.y), "f"(c.z), "f"(c.w));
    }
}

__global__ __launch_bounds__(256, 4) void diag_matmul_kernel_vec4_pow2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M4_shift,
    int total_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);
    for (int i = idx; i < total_vec4; i += stride) {
        int row = i >> M4_shift;
        float a = __ldg(&A[row]);
        float4 b;
        asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(b.x), "=f"(b.y), "=f"(b.z), "=f"(b.w)
            : "l"(B4 + i));
        float4 c;
        c.x = a * b.x;
        c.y = a * b.y;
        c.z = a * b.z;
        c.w = a * b.w;
        asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
            : : "l"(C4 + i), "f"(c.x), "f"(c.y), "f"(c.z), "f"(c.w));
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "Tensors must be float32");
    TORCH_CHECK(A.dim() == 1 && B.dim() == 2, "A must be 1D, B must be 2D");
    int N = A.size(0);
    int M = B.size(1);
    TORCH_CHECK(B.size(0) == N, "Leading dimensions must match");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");

    auto C = torch::empty({N, M}, B.options());

    const float* a_ptr = A.data_ptr<float>();
    const float* b_ptr = B.data_ptr<float>();
    float* c_ptr = C.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (M % 4 == 0) {
        int M4 = M / 4;
        int total = N * M4;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        blocks = min(blocks, 65535);
        bool is_pow2 = (M4 & (M4 - 1)) == 0;
        if (is_pow2) {
            int shift = __builtin_ctz(M4);
            diag_matmul_kernel_vec4_pow2<<<blocks, threads, 0, stream>>>(a_ptr, b_ptr, c_ptr, shift, total);
        } else {
            diag_matmul_kernel_vec4<<<blocks, threads, 0, stream>>>(a_ptr, b_ptr, c_ptr, N, M4);
        }
    } else {
        int total = N * M;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        blocks = min(blocks, 65535);
        diag_matmul_kernel<<<blocks, threads, 0, stream>>>(a_ptr, b_ptr, c_ptr, N, M);
    }

    return C;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
        C = diag(A) * B
        """
    def __init__(self):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        # <<<END_IMPROVE>>>

    def forward(self, A, B):
        # <<<IMPROVE:forward_stmt_1>>>
        if (A.is_cuda and B.is_cuda and
                    A.dtype == torch.float32 and B.dtype == torch.float32 and
                    A.dim() == 1 and B.dim() == 2 and
                    B.size(0) == A.size(0) and
                    A.is_contiguous() and B.is_contiguous()):
                    return _stark_get_extension().diag_matmul_cuda(A, B)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        return A.unsqueeze(1) * B
        # <<<END_IMPROVE>>>
