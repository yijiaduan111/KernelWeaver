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
    return f'stark_cuda_l2_p7_{digest}'

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

torch::Tensor fused_postops_cuda(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_postops", &fused_postops_cuda, "Fused ReLU->LeakyReLU->GELU->Sigmoid->Bias CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Fast path: applies all post-ops with a pre-loaded bias scalar.
// LeakyReLU is elided: ReLU guarantees v >= 0 on this path, so LeakyReLU(v) == v identically.
__device__ __forceinline__ float apply_postops_with_bias(float v, float b) {
    // ReLU
    v = v > 0.0f ? v : 0.0f;
    // LeakyReLU elided: after ReLU, v >= 0, so LeakyReLU is a mathematical identity.
    // GELU (tanh approximation matching PyTorch default)
    float v3 = v * v * v;
    float inner = 0.7978845608028654f * (v + 0.044715f * v3);
    v = 0.5f * v * (1.0f + tanhf(inner));
    // Sigmoid
    v =1.0f / (1.0f + expf(-v));
    // Bias add
    v += b;
    return v;
}

// General path: computes channel index and loads bias internally.
__device__ __forceinline__ float apply_postops(float v, const float* __restrict__ bias, int idx, int spatial, int C) {
    int channel = (idx / spatial) % C;
    return apply_postops_with_bias(v, bias[channel]);
}

__global__ void fused_postops_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int numel,
    int spatial,
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        out[idx] = apply_postops(x[idx], bias, idx, spatial, C);
    }
}

// Vec8 aligned kernel: valid only when spatial % 8 == 0.
// Each thread processes two consecutive float4 vectors; both always lie within
// the same channel slice, so we compute the channel index and load bias once
// for the pair, then apply post-ops to all 8 lanes.
__global__ void fused_postops_kernel_vec8_aligned(
    const float4* __restrict__ x,
    const float* __restrict__ bias,
    float4* __restrict__ out,
    int nvec,        // total float4 vectors = numel/4
    int spatial_vec, // spatial / 4
    int C
) {
    // Each thread handles a pair of float4 vectors (indices 2*tid, 2*tid+1)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int npairs = nvec / 2;
    for (int pair = tid; pair < npairs; pair += stride) {
        int vi0 = pair * 2;
        int vi1 = vi0 + 1;
        // Both vectors lie in the same channel when spatial%8==0
        int channel = (vi0 / spatial_vec) % C;
        float b = bias[channel];
        float4 v0 = x[vi0];
        float4 v1 = x[vi1];
        v0.x = apply_postops_with_bias(v0.x, b);
        v0.y = apply_postops_with_bias(v0.y, b);
        v0.z = apply_postops_with_bias(v0.z, b);
        v0.w = apply_postops_with_bias(v0.w, b);
        v1.x = apply_postops_with_bias(v1.x, b);
        v1.y = apply_postops_with_bias(v1.y, b);
        v1.z = apply_postops_with_bias(v1.z, b);
        v1.w = apply_postops_with_bias(v1.w, b);
        out[vi0] = v0;
        out[vi1] = v1;
    }
}

// Aligned-channel vec4 kernel: valid only when spatial % 4 == 0.
// In that case every float4 lies entirely within one channel slice,
// so we compute channel once per vector and load bias once.
__global__ void fused_postops_kernel_vec4_aligned(
    const float4* __restrict__ x,
    const float* __restrict__ bias,
    float4* __restrict__ out,
    int nvec,
    int spatial_vec,  // spatial / 4
    int C
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; vec_idx < nvec; vec_idx += stride) {
        // Each vec_idx maps to exactly one channel (no boundary crossing)
        int channel = (vec_idx / spatial_vec) % C;
        float b = bias[channel];
        float4 v4 = x[vec_idx];
        v4.x = apply_postops_with_bias(v4.x, b);
        v4.y = apply_postops_with_bias(v4.y, b);
        v4.z = apply_postops_with_bias(v4.z, b);
        v4.w = apply_postops_with_bias(v4.w, b);
        out[vec_idx] = v4;
    }
}

// Mixed-channel vec4 kernel: used when spatial % 4 != 0 but numel % 4 == 0.
__global__ void fused_postops_kernel_vec4(
    const float4* __restrict__ x,
    const float* __restrict__ bias,
    float4* __restrict__ out,
    int nvec,
    int spatial,
    int C
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; vec_idx < nvec; vec_idx += stride) {
        int base = vec_idx * 4;
        float4 v4 = x[vec_idx];
        int c0 = (base / spatial) % C;
        int c3 = ((base + 3) / spatial) % C;
        if (c0 == c3) {
            float b = bias[c0];
            v4.x = apply_postops_with_bias(v4.x, b);
            v4.y = apply_postops_with_bias(v4.y, b);
            v4.z = apply_postops_with_bias(v4.z, b);
            v4.w = apply_postops_with_bias(v4.w, b);
        } else {
            v4.x = apply_postops(v4.x, bias, base + 0, spatial, C);
            v4.y = apply_postops(v4.y, bias, base + 1, spatial, C);
            v4.z = apply_postops(v4.z, bias, base + 2, spatial, C);
            v4.w = apply_postops(v4.w, bias, base + 3, spatial, C);
        }
        out[vec_idx] = v4;
    }
}

torch::Tensor fused_postops_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,C,D,H,W)");

    auto out = torch::empty_like(x);
    int numel = (int)x.numel();
    int C = (int)x.size(1);
    int spatial = (int)(x.size(2) * x.size(3) * x.size(4));

    const int threads = 256;

    bool aligned = (numel % 4 == 0)
        && ((reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16) == 0)
        && ((reinterpret_cast<uintptr_t>(out.data_ptr<float>()) % 16) == 0);

    if (aligned) {
        int nvec = numel / 4;
        if (spatial % 8 == 0 && nvec % 2 == 0) {
            // Vec8 path: two float4 per thread, both in same channel
            int spatial_vec = spatial / 4;
            int npairs = nvec / 2;
            int blocks = (npairs + threads - 1) / threads;
            fused_postops_kernel_vec8_aligned<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(x.data_ptr<float>()),
                bias.data_ptr<float>(),
                reinterpret_cast<float4*>(out.data_ptr<float>()),
                nvec,
                spatial_vec,
                C
            );
        } else if (spatial % 4 == 0) {
            // Branch-free aligned-channel path: every float4 stays within one channel
            int spatial_vec = spatial / 4;
            int blocks = (nvec + threads - 1) / threads;
            fused_postops_kernel_vec4_aligned<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(x.data_ptr<float>()),
                bias.data_ptr<float>(),
                reinterpret_cast<float4*>(out.data_ptr<float>()),
                nvec,
                spatial_vec,
                C
            );
        } else {
            // Mixed-channel path with per-vector boundary check
            int blocks = (nvec + threads - 1) / threads;
            fused_postops_kernel_vec4<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(x.data_ptr<float>()),
                bias.data_ptr<float>(),
                reinterpret_cast<float4*>(out.data_ptr<float>()),
                nvec,
                spatial,
                C
            );
        }
    } else {
        int blocks = (numel + threads - 1) / threads;
        fused_postops_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            numel,
            spatial,
            C
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
        """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_postops(x.contiguous(), self.bias.contiguous())
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # leaky_relu handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # gelu handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # sigmoid handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # bias add handled inside fused CUDA epilogue
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
