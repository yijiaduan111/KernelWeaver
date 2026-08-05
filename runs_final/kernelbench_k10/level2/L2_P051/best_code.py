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
    return f'stark_cuda_l2_p51_{digest}'

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

torch::Tensor rowdot_gelu_residual_cuda(torch::Tensor x, torch::Tensor reduced_w, torch::Tensor reduced_bias);

torch::Tensor rowdot_gelu_residual(torch::Tensor x, torch::Tensor reduced_w, torch::Tensor reduced_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(reduced_w.is_cuda(), "reduced_w must be a CUDA tensor");
    TORCH_CHECK(reduced_bias.is_cuda(), "reduced_bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(reduced_w.is_contiguous(), "reduced_w must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    return rowdot_gelu_residual_cuda(x, reduced_w, reduced_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowdot_gelu_residual", &rowdot_gelu_residual, "Row dot + GELU + residual add (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 512

__device__ __forceinline__ float gelu_approx(float val) {
    return 0.5f * val * (1.0f + erff(val * 0.7071067811865476f));
}

// Specialized kernel for feat==8192 with guaranteed 16-byte alignment.
// No runtime alignment checks, no branching on feat, pure unrolled float4 path.
__global__ void rowdot_gelu_residual_kernel_8192(
    const float* __restrict__ x,
    const float* __restrict__ reduced_w,
    const float* __restrict__ reduced_bias,
    float* __restrict__ out,
    int batch
) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const float4* x_vec = reinterpret_cast<const float4*>(x + row * 8192);
    const float4* w_vec = reinterpret_cast<const float4*>(reduced_w);
    float4* out_vec = reinterpret_cast<float4*>(out + row * 8192);

    int j = threadIdx.x;

    // Each thread loads 4 float4 elements (16 floats) covering 8192/4=2048 float4 lanes
    // with BLOCK_SIZE=512: 4 * 512 = 2048 exactly.
    float4 xv0 = x_vec[j];
    float4 wv0 = w_vec[j];
    float4 xv1 = x_vec[j + BLOCK_SIZE];
    float4 wv1 = w_vec[j + BLOCK_SIZE];
    float4 xv2 = x_vec[j + 2*BLOCK_SIZE];
    float4 wv2 = w_vec[j + 2*BLOCK_SIZE];
    float4 xv3 = x_vec[j + 3*BLOCK_SIZE];
    float4 wv3 = w_vec[j + 3*BLOCK_SIZE];

    float local_sum = 0.0f;
    local_sum += xv0.x*wv0.x + xv0.y*wv0.y + xv0.z*wv0.z + xv0.w*wv0.w;
    local_sum += xv1.x*wv1.x + xv1.y*wv1.y + xv1.z*wv1.z + xv1.w*wv1.w;
    local_sum += xv2.x*wv2.x + xv2.y*wv2.y + xv2.z*wv2.z + xv2.w*wv2.w;
    local_sum += xv3.x*wv3.x + xv3.y*wv3.y + xv3.z*wv3.z + xv3.w*wv3.w;

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    __shared__ float warp_sums[16];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0)
        warp_sums[warp_id] = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float dot_val = 0.0f;
        dot_val += warp_sums[0]  + warp_sums[1]  + warp_sums[2]  + warp_sums[3];
        dot_val += warp_sums[4]  + warp_sums[5]  + warp_sums[6]  + warp_sums[7];
        dot_val += warp_sums[8]  + warp_sums[9]  + warp_sums[10] + warp_sums[11];
        dot_val += warp_sums[12] + warp_sums[13] + warp_sums[14] + warp_sums[15];
        dot_val += reduced_bias[0];
        warp_sums[0] = gelu_approx(dot_val);
    }
    __syncthreads();

    float scalar = warp_sums[0];

    // Write back using already-loaded x values (register reuse, no reload)
    float4 r0 = {xv0.x+scalar, xv0.y+scalar, xv0.z+scalar, xv0.w+scalar};
    float4 r1 = {xv1.x+scalar, xv1.y+scalar, xv1.z+scalar, xv1.w+scalar};
    float4 r2 = {xv2.x+scalar, xv2.y+scalar, xv2.z+scalar, xv2.w+scalar};
    float4 r3 = {xv3.x+scalar, xv3.y+scalar, xv3.z+scalar, xv3.w+scalar};
    out_vec[j]              = r0;
    out_vec[j + BLOCK_SIZE]   = r1;
    out_vec[j + 2*BLOCK_SIZE] = r2;
    out_vec[j + 3*BLOCK_SIZE] = r3;
}

// Generic fallback kernel for other feat sizes.
__global__ void rowdot_gelu_residual_kernel_generic(
    const float* __restrict__ x,
    const float* __restrict__ reduced_w,
    const float* __restrict__ reduced_bias,
    float* __restrict__ out,
    int batch,
    int feat
) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const float* x_row = x + row * feat;
    float* out_row = out + row * feat;

    bool aligned = (reinterpret_cast<uintptr_t>(x_row) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(reduced_w) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(out_row) % 16 == 0);

    float local_sum = 0.0f;

    if ((feat % 4 == 0) && aligned) {
        const float4* x_vec = reinterpret_cast<const float4*>(x_row);
        const float4* w_vec = reinterpret_cast<const float4*>(reduced_w);
        int feat4 = feat >> 2;
        for (int j = threadIdx.x; j < feat4; j += BLOCK_SIZE) {
            float4 xv = x_vec[j];
            float4 wv = w_vec[j];
            local_sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    } else {
        for (int j = threadIdx.x; j < feat; j += BLOCK_SIZE) {
            local_sum += x_row[j] * reduced_w[j];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    __shared__ float warp_sums[16];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0)
        warp_sums[warp_id] = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float dot_val = 0.0f;
        for (int w = 0; w < 16; w++)
            dot_val += warp_sums[w];
        dot_val += reduced_bias[0];
        warp_sums[0] = gelu_approx(dot_val);
    }
    __syncthreads();

    float scalar = warp_sums[0];

    if ((feat % 4 == 0) && aligned) {
        const float4* x_vec = reinterpret_cast<const float4*>(x_row);
        float4* out_vec = reinterpret_cast<float4*>(out_row);
        int feat4 = feat >> 2;
        for (int j = threadIdx.x; j < feat4; j += BLOCK_SIZE) {
            float4 xv = x_vec[j];
            float4 result = {xv.x+scalar, xv.y+scalar, xv.z+scalar, xv.w+scalar};
            out_vec[j] = result;
        }
    } else {
        for (int j = threadIdx.x; j < feat; j += BLOCK_SIZE) {
            out_row[j] = x_row[j] + scalar;
        }
    }
}

torch::Tensor rowdot_gelu_residual_cuda(
    torch::Tensor x,
    torch::Tensor reduced_w,
    torch::Tensor reduced_bias
) {
    int batch = x.size(0);
    int feat  = x.size(1);

    auto out = torch::empty_like(x);

    dim3 grid(batch);
    dim3 block(BLOCK_SIZE);

    // Check alignment on host side for the specialized 8192 kernel.
    bool aligned = (reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(reduced_w.data_ptr<float>()) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(out.data_ptr<float>()) % 16 == 0);

    if (feat == 8192 && aligned) {
        rowdot_gelu_residual_kernel_8192<<<grid, block>>>(
            x.data_ptr<float>(),
            reduced_w.data_ptr<float>(),
            reduced_bias.data_ptr<float>(),
            out.data_ptr<float>(),
            batch
        );
    } else {
        rowdot_gelu_residual_kernel_generic<<<grid, block>>>(
            x.data_ptr<float>(),
            reduced_w.data_ptr<float>(),
            reduced_bias.data_ptr<float>(),
            out.data_ptr<float>(),
            batch,
            feat
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
        """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        original_x = x
        if self.gemm.bias is not None:
            reduced_bias = ((self.gemm.bias - self.subtract).mean()).view(1).contiguous()
        else:
            reduced_bias = (-self.subtract.mean()).view(1).contiguous()
        # reduced_w: mean over output dim of weight, shape [in_features]
        # nn.Linear weight is [out_features, in_features], so mean(dim=0) gives [in_features]
        # The mean result = dot(x[row], reduced_w) + reduced_bias (already divided by out_features)
        reduced_w = (self.gemm.weight.mean(dim=0)).contiguous()
        return _stark_get_extension().rowdot_gelu_residual(x.contiguous(), reduced_w, reduced_bias)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        # collapsed into CUDA fast path above; unreachable
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # collapsed into CUDA fast path above; unreachable
        x = x - self.subtract
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # collapsed into CUDA fast path above; unreachable
        x = torch.mean(x, dim=1, keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = torch.logsumexp(x, dim=1, keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        x = torch.nn.functional.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        x = x + original_x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        return x
        # <<<END_IMPROVE>>>
