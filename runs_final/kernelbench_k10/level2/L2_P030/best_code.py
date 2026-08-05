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
    return f'stark_cuda_l2_p30_{digest}'

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

torch::Tensor fused_groupnorm_hardtanh(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    double eps,
    double min_val,
    double max_val
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_groupnorm_hardtanh", &fused_groupnorm_hardtanh,
          "Fused GroupNorm + HardTanh (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Vectorized fast-path kernel: group_size == 512, 128 threads, each thread
// handles 4 contiguous elements via float4 loads/stores.
// Address arithmetic hoisted: row pointers computed once per block.
// Register-pressure reduction: input values NOT kept alive across reduction.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(128, 4)
void fused_groupnorm_hardtanh_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int num_groups,
    float inv_group_size,
    float eps,
    float min_val,
    float max_val
) {
    const int group_size = 512;
    int batch_idx = blockIdx.x / num_groups;
    int group_idx = blockIdx.x % num_groups;

    // Hoist row pointers once; avoids repeated multiply inside loops
    const float* __restrict__ in_row  = input  + batch_idx * num_groups * group_size + group_idx * group_size;
          float* __restrict__ out_row = output + batch_idx * num_groups * group_size + group_idx * group_size;
    const float* __restrict__ w_row   = weight + group_idx * group_size;
    const float* __restrict__ b_row   = bias   + group_idx * group_size;

    int elem = threadIdx.x * 4;  // 0, 4, 8, ... 508

    // --- Phase 1: load for stats only; let registers die at end of scope ---
    float local_sum, local_sq;
    {
        float4 v = *reinterpret_cast<const float4*>(in_row + elem);
        local_sum = v.x + v.y + v.z + v.w;
        local_sq  = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    // Warp-level reduction
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
        local_sq  += __shfl_down_sync(mask, local_sq,  offset);
    }

    // 128 threads => 4 warps; store warp partials to shared memory
    extern __shared__ float smem[];  // 8 floats: [0..3]=sum, [4..7]=sumsq
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    if (lane_id == 0) {
        smem[warp_id]     = local_sum;
        smem[warp_id + 4] = local_sq;
    }
    __syncthreads();

    // First warp reduces 4 warp partials
    if (warp_id == 0) {
        float ws = (lane_id < 4) ? smem[lane_id]     : 0.0f;
        float wq = (lane_id < 4) ? smem[lane_id + 4] : 0.0f;
        for (int offset = 2; offset > 0; offset >>= 1) {
            ws += __shfl_down_sync(mask, ws, offset);
            wq += __shfl_down_sync(mask, wq, offset);
        }
        if (lane_id == 0) {
            float mean = ws * inv_group_size;
            float var  = wq * inv_group_size - mean * mean;
            smem[0] = mean;
            smem[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    float mean    = smem[0];
    float inv_std = smem[1];

    // --- Phase 2: reload via hoisted pointers, normalize, clamp, store ---
    float4 in2 = *reinterpret_cast<const float4*>(in_row + elem);
    float4 w4  = *reinterpret_cast<const float4*>(w_row  + elem);
    float4 b4  = *reinterpret_cast<const float4*>(b_row  + elem);

    float4 out4;
    out4.x = fminf(fmaxf((in2.x - mean) * inv_std * w4.x + b4.x, min_val), max_val);
    out4.y = fminf(fmaxf((in2.y - mean) * inv_std * w4.y + b4.y, min_val), max_val);
    out4.z = fminf(fmaxf((in2.z - mean) * inv_std * w4.z + b4.z, min_val), max_val);
    out4.w = fminf(fmaxf((in2.w - mean) * inv_std * w4.w + b4.w, min_val), max_val);

    *reinterpret_cast<float4*>(out_row + elem) = out4;
}

// ---------------------------------------------------------------------------
// Generic scalar fallback kernel.
// ---------------------------------------------------------------------------
__global__ void fused_groupnorm_hardtanh_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int num_groups,
    int group_size,
    float eps,
    float min_val,
    float max_val
) {
    int batch_idx = blockIdx.x / num_groups;
    int group_idx = blockIdx.x % num_groups;

    // Hoist row pointers
    const float* __restrict__ in_row  = input  + batch_idx * num_groups * group_size + group_idx * group_size;
          float* __restrict__ out_row = output + batch_idx * num_groups * group_size + group_idx * group_size;
    const float* __restrict__ w_row   = weight + group_idx * group_size;
    const float* __restrict__ b_row   = bias   + group_idx * group_size;

    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = in_row[i];
        local_sum += v;
        local_sq  += v * v;
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
        local_sq  += __shfl_down_sync(mask, local_sq,  offset);
    }

    extern __shared__ float smem[];
    int warp_id   = threadIdx.x >> 5;
    int lane_id   = threadIdx.x & 31;
    int num_warps = (blockDim.x + 31) >> 5;
    float* smem_sum = smem;
    float* smem_sq  = smem + num_warps;

    if (lane_id == 0) {
        smem_sum[warp_id] = local_sum;
        smem_sq [warp_id] = local_sq;
    }
    __syncthreads();

    float block_sum = 0.0f, block_sq = 0.0f;
    if (threadIdx.x < num_warps) {
        block_sum = smem_sum[threadIdx.x];
        block_sq  = smem_sq [threadIdx.x];
    }
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
            block_sq  += __shfl_down_sync(mask, block_sq,  offset);
        }
    }

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = block_sum / group_size;
        float var  = block_sq  / group_size - mean * mean;
        s_mean     = mean;
        s_inv_std  = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean    = s_mean;
    float inv_std = s_inv_std;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v   = in_row[i];
        float out = (v - mean) * inv_std * w_row[i] + b_row[i];
        out_row[i] = fminf(fmaxf(out, min_val), max_val);
    }
}

// ---------------------------------------------------------------------------
// Host wrapper.
// ---------------------------------------------------------------------------
torch::Tensor fused_groupnorm_hardtanh(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    double eps,
    double min_val,
    double max_val
) {
    TORCH_CHECK(input.is_cuda(),       "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int batch_size   = input.size(0);
    int out_features = input.size(1);
    int group_size   = out_features / num_groups;
    int grid_size    = batch_size * num_groups;

    auto output = torch::empty_like(input);

    const float feps     = static_cast<float>(eps);
    const float fmin_val = static_cast<float>(min_val);
    const float fmax_val = static_cast<float>(max_val);

    bool use_vec4 = (group_size == 512) &&
                    ((reinterpret_cast<uintptr_t>(input.data_ptr<float>())  & 15) == 0) &&
                    ((reinterpret_cast<uintptr_t>(output.data_ptr<float>()) & 15) == 0) &&
                    ((reinterpret_cast<uintptr_t>(weight.data_ptr<float>()) & 15) == 0) &&
                    ((reinterpret_cast<uintptr_t>(bias.data_ptr<float>())   & 15) == 0);

    if (use_vec4) {
        const int    block_size = 128;
        const size_t smem_bytes = 8 * sizeof(float);
        fused_groupnorm_hardtanh_vec4_kernel<<<grid_size, block_size, smem_bytes>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            num_groups,
            1.0f / static_cast<float>(group_size),
            feps,
            fmin_val,
            fmax_val
        );
    } else {
        int block_size = group_size;
        if (block_size > 1024) block_size = 1024;
        block_size = ((block_size + 31) / 32) * 32;
        if (block_size > 1024) block_size = 1024;
        int    num_warps  = (block_size + 31) / 32;
        size_t smem_bytes = 2 * num_warps * sizeof(float);

        fused_groupnorm_hardtanh_scalar_kernel<<<grid_size, block_size, smem_bytes>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            num_groups,
            group_size,
            feps,
            fmin_val,
            fmax_val
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
        """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
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
        x = self.gemm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().fused_groupnorm_hardtanh(
            x.contiguous(),
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
            self.hardtanh.min_val,
            self.hardtanh.max_val,
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # HardTanh is fused into the CUDA kernel above; nothing to do here.
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
