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
    return f'stark_cuda_l2_p95_{digest}'

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

torch::Tensor fused_post_ops(torch::Tensor x, torch::Tensor add_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_ops", &fused_post_ops, "Fused add+swish+tanh+gelu+hardtanh (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_post_ops_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add_value,
    float* __restrict__ out,
    int N,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int col = idx % N;
    float v = x[idx] + __ldg(&add_value[col]);

    // Swish: v * sigmoid(v)
    float sig = 1.0f / (1.0f + __expf(-v));
    v = v * sig;

    // Tanh
    v = tanhf(v);

    // GELU (tanh approximation matching PyTorch default)
    float v3 = v * v * v;
    float inner = 0.7978845608028654f * (v + 0.044715f * v3);
    v = 0.5f * v * (1.0f + tanhf(inner));

    // Hardtanh [-1, 1]
    v = fmaxf(-1.0f, fminf(1.0f, v));

    out[idx] = v;
}

__global__ void fused_post_ops_vec4_kernel(
    const float4* __restrict__ x,
    const float* __restrict__ add_value,
    float4* __restrict__ out,
    int N,
    int total4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total4) return;

    int base = idx * 4;
    float4 v4 = x[idx];
    float vals[4] = {v4.x, v4.y, v4.z, v4.w};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int col = (base + i) % N;
        float v = vals[i] + __ldg(&add_value[col]);

        // Swish
        float sig = 1.0f / (1.0f + __expf(-v));
        v = v * sig;

        // Tanh
        v = tanhf(v);

        // GELU (tanh approximation)
        float v3 = v * v * v;
        float inner = 0.7978845608028654f * (v + 0.044715f * v3);
        v = 0.5f * v * (1.0f + tanhf(inner));

        // Hardtanh [-1, 1]
        v = fmaxf(-1.0f, fminf(1.0f, v));

        vals[i] = v;
    }

    float4 res;
    res.x = vals[0]; res.y = vals[1]; res.z = vals[2]; res.w = vals[3];
    out[idx] = res;
}

torch::Tensor fused_post_ops(torch::Tensor x, torch::Tensor add_value) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(add_value.is_cuda(), "add_value must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(add_value.scalar_type() == torch::kFloat32, "add_value must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(add_value.is_contiguous(), "add_value must be contiguous");

    int total = x.numel();
    int N = x.size(x.dim() - 1);
    auto out = torch::empty_like(x);

    bool use_vec4 = (total % 4 == 0) && (N % 4 == 0) &&
                    ((reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16) == 0) &&
                    ((reinterpret_cast<uintptr_t>(out.data_ptr<float>()) % 16) == 0);

    const int threads = 256;
    if (use_vec4) {
        int total4 = total / 4;
        int blocks = (total4 + threads - 1) / threads;
        fused_post_ops_vec4_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            add_value.data_ptr<float>(),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            N,
            total4
        );
    } else {
        int blocks = (total + threads - 1) / threads;
        fused_post_ops_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            add_value.data_ptr<float>(),
            out.data_ptr<float>(),
            N,
            total
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
        """
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.matmul(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_post_ops(x.contiguous(), self.add_value.contiguous())
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # swish fused in fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # tanh fused in fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # gelu fused in fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # hardtanh fused in fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
