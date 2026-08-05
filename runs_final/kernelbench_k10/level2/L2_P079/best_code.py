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
    return f'stark_cuda_l2_p79_{digest}'

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

torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor multiplier,
                                   double eps, double clamp_min, double clamp_max);

torch::Tensor fused_post_conv(torch::Tensor x, torch::Tensor multiplier,
                              double eps, double clamp_min, double clamp_max) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "multiplier must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    return fused_post_conv_cuda(x, multiplier.contiguous(), eps, clamp_min, clamp_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused multiply+instancenorm+clamp+multiply+max (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Pass 1: compute per-(n,c) mean and inv_std over D*H*W elements.
// Reads raw x and applies per-channel multiplier inline before accumulating stats.
// 128-thread blocks with min 2 blocks/SM to reduce register pressure.
__global__ __launch_bounds__(128, 2)
void stats_kernel_inline(
    const float* __restrict__ x,
    const float* __restrict__ multiplier,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    int C, int DHW, float eps
) {
    int nc  = blockIdx.x;
    int c   = nc % C;
    float m = multiplier[c];
    const float* slice = x + (long)nc * DHW;

    float sum = 0.f, sum_sq = 0.f;
    for (int i = threadIdx.x; i < DHW; i += blockDim.x) {
        float v = slice[i] * m;
        sum    += v;
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum    += __shfl_xor_sync(0xffffffff, sum,    offset);
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    extern __shared__ float smem[];
    int lane   = threadIdx.x & 31;
    int wid    = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    if (lane == 0) {
        smem[wid]          = sum;
        smem[wid + nwarps] = sum_sq;
    }
    __syncthreads();

    if (threadIdx.x < nwarps) {
        sum    = smem[threadIdx.x];
        sum_sq = smem[threadIdx.x + nwarps];
    } else {
        sum    = 0.f;
        sum_sq = 0.f;
    }
    if (threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum    += __shfl_xor_sync(0xffffffff, sum,    offset);
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
    }

    if (threadIdx.x == 0) {
        float mean = sum / (float)DHW;
        float var  = sum_sq / (float)DHW - mean * mean;
        mean_out[nc]    = mean;
        inv_std_out[nc] = rsqrtf(var + eps);
    }
}

// Pass 2 generic: multiply inline, normalize, clamp, multiply again, then channel-max.
// 128-thread blocks with min 2 blocks/SM.
__global__ __launch_bounds__(128, 2)
void normalize_clamp_max_kernel_inline(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ multiplier,
    float* __restrict__ out,
    int N, int C, int DHW,
    float clamp_min, float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * DHW;
    if (idx >= total) return;

    int n   = idx / DHW;
    int dhw = idx % DHW;

    float max_val = -1e38f;
    for (int c = 0; c < C; c++) {
        int nc = n * C + c;
        float m = multiplier[c];
        float v = x[(long)nc * DHW + dhw] * m;
        v = (v - mean[nc]) * inv_std[nc];
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        v = v * m;
        max_val = fmaxf(max_val, v);
    }
    out[idx] = max_val;
}

// Pass 2 specialized for C==16: per-sample tiled mapping with precomputed affine constants.
// Scalar path used unconditionally; float4 path taken when DHW%4==0 and pointers are
// 16-byte aligned (checked at launch in the wrapper).
__global__ __launch_bounds__(128, 2)
void normalize_clamp_max_kernel_c16_tiled(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ multiplier,
    float* __restrict__ out,
    int DHW,
    float clamp_min, float clamp_max
) {
    int n   = blockIdx.y;
    int dhw = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_multiplier[16];
    __shared__ float s_norm_scale[16];   // multiplier[c] * inv_std[n*16+c]
    __shared__ float s_norm_bias[16];    // -mean[n*16+c] * inv_std[n*16+c]

    if (threadIdx.x < 16) {
        float m   = multiplier[threadIdx.x];
        int nc    = n * 16 + threadIdx.x;
        float mu  = mean[nc];
        float is  = inv_std[nc];
        s_multiplier[threadIdx.x] = m;
        s_norm_scale[threadIdx.x] = m * is;
        s_norm_bias[threadIdx.x]  = -mu * is;
    }
    __syncthreads();

    if (dhw >= DHW) return;

    int base_nc = n * 16;

    #define PROCESS_C(ci) \
        float v##ci = x[(long)(base_nc + ci) * DHW + dhw] * s_norm_scale[ci] + s_norm_bias[ci]; \
        v##ci = fmaxf(clamp_min, fminf(clamp_max, v##ci)); \
        v##ci = v##ci * s_multiplier[ci];

    PROCESS_C(0)
    PROCESS_C(1)
    PROCESS_C(2)
    PROCESS_C(3)
    PROCESS_C(4)
    PROCESS_C(5)
    PROCESS_C(6)
    PROCESS_C(7)
    PROCESS_C(8)
    PROCESS_C(9)
    PROCESS_C(10)
    PROCESS_C(11)
    PROCESS_C(12)
    PROCESS_C(13)
    PROCESS_C(14)
    PROCESS_C(15)

    #undef PROCESS_C

    float max_val = fmaxf(fmaxf(fmaxf(fmaxf(v0,  v1),  fmaxf(v2,  v3)),
                                fmaxf(fmaxf(v4,  v5),  fmaxf(v6,  v7))),
                          fmaxf(fmaxf(fmaxf(v8,  v9),  fmaxf(v10, v11)),
                                fmaxf(fmaxf(v12, v13), fmaxf(v14, v15))));
    out[n * DHW + dhw] = max_val;
}

// Pass 2 specialized for C==16, vectorized float4 path.
// Each thread processes 4 contiguous DHW positions.
// Only launched when DHW%4==0 and pointers are 16-byte aligned.
__global__ __launch_bounds__(128, 2)
void normalize_clamp_max_kernel_c16_vec4(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ multiplier,
    float* __restrict__ out,
    int DHW,
    float clamp_min, float clamp_max
) {
    int n    = blockIdx.y;
    int dhw4 = blockIdx.x * blockDim.x + threadIdx.x;  // index into DHW/4 chunks

    __shared__ float s_multiplier[16];
    __shared__ float s_norm_scale[16];
    __shared__ float s_norm_bias[16];

    if (threadIdx.x < 16) {
        float m   = multiplier[threadIdx.x];
        int nc    = n * 16 + threadIdx.x;
        float mu  = mean[nc];
        float is  = inv_std[nc];
        s_multiplier[threadIdx.x] = m;
        s_norm_scale[threadIdx.x] = m * is;
        s_norm_bias[threadIdx.x]  = -mu * is;
    }
    __syncthreads();

    int DHW4 = DHW >> 2;  // DHW / 4
    if (dhw4 >= DHW4) return;

    int base_nc = n * 16;

    // Accumulators: max over 16 channels for each of 4 lanes
    float mx0 = -1e38f, mx1 = -1e38f, mx2 = -1e38f, mx3 = -1e38f;

    #define PROCESS_C4(ci) \
    { \
        const float4 xv = reinterpret_cast<const float4*>(x + (long)(base_nc + ci) * DHW)[dhw4]; \
        float ns = s_norm_scale[ci]; \
        float nb = s_norm_bias[ci]; \
        float mp = s_multiplier[ci]; \
        float r0 = fmaxf(clamp_min, fminf(clamp_max, xv.x * ns + nb)) * mp; \
        float r1 = fmaxf(clamp_min, fminf(clamp_max, xv.y * ns + nb)) * mp; \
        float r2 = fmaxf(clamp_min, fminf(clamp_max, xv.z * ns + nb)) * mp; \
        float r3 = fmaxf(clamp_min, fminf(clamp_max, xv.w * ns + nb)) * mp; \
        mx0 = fmaxf(mx0, r0); \
        mx1 = fmaxf(mx1, r1); \
        mx2 = fmaxf(mx2, r2); \
        mx3 = fmaxf(mx3, r3); \
    }

    PROCESS_C4(0)
    PROCESS_C4(1)
    PROCESS_C4(2)
    PROCESS_C4(3)
    PROCESS_C4(4)
    PROCESS_C4(5)
    PROCESS_C4(6)
    PROCESS_C4(7)
    PROCESS_C4(8)
    PROCESS_C4(9)
    PROCESS_C4(10)
    PROCESS_C4(11)
    PROCESS_C4(12)
    PROCESS_C4(13)
    PROCESS_C4(14)
    PROCESS_C4(15)

    #undef PROCESS_C4

    float4 res;
    res.x = mx0; res.y = mx1; res.z = mx2; res.w = mx3;
    reinterpret_cast<float4*>(out + n * DHW)[dhw4] = res;
}

torch::Tensor fused_post_conv_cuda(
    torch::Tensor x,
    torch::Tensor multiplier,
    double eps,
    double clamp_min,
    double clamp_max
) {
    int N   = x.size(0);
    int C   = x.size(1);
    int D   = x.size(2);
    int H   = x.size(3);
    int W   = x.size(4);
    int DHW = D * H * W;

    auto mul_flat = multiplier.view({C}).contiguous();
    auto opts     = x.options();

    auto mean_buf    = torch::empty({N * C}, opts);
    auto inv_std_buf = torch::empty({N * C}, opts);

    int block1     = 128;
    int nwarps1    = (block1 + 31) / 32;
    int smem_bytes = 2 * nwarps1 * (int)sizeof(float);
    stats_kernel_inline<<<N * C, block1, smem_bytes>>>(
        x.data_ptr<float>(),
        mul_flat.data_ptr<float>(),
        mean_buf.data_ptr<float>(),
        inv_std_buf.data_ptr<float>(),
        C, DHW, (float)eps
    );

    auto out   = torch::empty({N, D, H, W}, opts);
    int block2 = 128;

    if (C == 16) {
        // Use vectorized float4 path when DHW is divisible by 4 and base pointers
        // are 16-byte aligned (PyTorch CUDA allocator guarantees >= 256-byte alignment).
        bool use_vec4 = (DHW % 4 == 0) &&
                        ((reinterpret_cast<uintptr_t>(x.data_ptr<float>()) & 15) == 0) &&
                        ((reinterpret_cast<uintptr_t>(out.data_ptr<float>()) & 15) == 0);
        if (use_vec4) {
            int DHW4 = DHW / 4;
            dim3 grid2((DHW4 + block2 - 1) / block2, N);
            normalize_clamp_max_kernel_c16_vec4<<<grid2, block2>>>(
                x.data_ptr<float>(),
                mean_buf.data_ptr<float>(),
                inv_std_buf.data_ptr<float>(),
                mul_flat.data_ptr<float>(),
                out.data_ptr<float>(),
                DHW,
                (float)clamp_min, (float)clamp_max
            );
        } else {
            dim3 grid2((DHW + block2 - 1) / block2, N);
            normalize_clamp_max_kernel_c16_tiled<<<grid2, block2>>>(
                x.data_ptr<float>(),
                mean_buf.data_ptr<float>(),
                inv_std_buf.data_ptr<float>(),
                mul_flat.data_ptr<float>(),
                out.data_ptr<float>(),
                DHW,
                (float)clamp_min, (float)clamp_max
            );
        }
    } else {
        int total = N * DHW;
        int grid2 = (total + block2 - 1) / block2;
        normalize_clamp_max_kernel_inline<<<grid2, block2>>>(
            x.data_ptr<float>(),
            mean_buf.data_ptr<float>(),
            inv_std_buf.data_ptr<float>(),
            mul_flat.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, DHW,
            (float)clamp_min, (float)clamp_max
        );
    }

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_post_conv(x.contiguous(), self.multiplier, 1e-5, float(self.clamp_min), float(self.clamp_max))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        # fused into fused_post_conv
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # fused into fused_post_conv
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # fused into fused_post_conv
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # fused into fused_post_conv; output already has shape [N,D,H,W]
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return x
        # <<<END_IMPROVE>>>
