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
    return f'stark_cuda_l2_p96_{digest}'

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

torch::Tensor fused_scale_maxpool_globalavgpool_clamp_cuda(torch::Tensor x, double scale, double clamp_min, double clamp_max, int64_t pool_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scale_maxpool_globalavgpool_clamp_cuda", &fused_scale_maxpool_globalavgpool_clamp_cuda, "Fused scale+maxpool+globalavgpool+clamp (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_scale_maxpool_globalavgpool_clamp_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int D, int H, int W,
    int pool_k,
    int Dp, int Hp, int Wp,
    float scale,
    float clamp_min, float clamp_max
) {
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;

    if (n >= N || c >= C) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* x_nc = x + n * (C * D * H * W) + c * (D * H * W);

    int total_pooled = Dp * Hp * Wp;
    int HpWp = Hp * Wp;

    float thread_sum = 0.0f;

    if (pool_k == 2) {
        int HW = H * W;
        for (int pool_idx = tid; pool_idx < total_pooled; pool_idx += block_size) {
            int dp = pool_idx / HpWp;
            int hp = (pool_idx - dp * HpWp) / Wp;
            int wp = pool_idx - dp * HpWp - hp * Wp;

            int d0 = dp << 1;
            int h0 = hp << 1;
            int w0 = wp << 1;

            const float* row0 = x_nc + d0 * HW + h0 * W + w0;
            const float* row1 = row0 + W;
            const float* row2 = row0 + HW;
            const float* row3 = row2 + W;

            // Only take the float2 fast path when every row is 8-byte aligned.
            bool aligned = (((reinterpret_cast<uintptr_t>(row0) |
                              reinterpret_cast<uintptr_t>(row1) |
                              reinterpret_cast<uintptr_t>(row2) |
                              reinterpret_cast<uintptr_t>(row3)) & 7ULL) == 0ULL);

            float v00, v01, v10, v11, v20, v21, v30, v31;

            if (aligned) {
                const float2* row0_f2 = reinterpret_cast<const float2*>(row0);
                const float2* row1_f2 = reinterpret_cast<const float2*>(row1);
                const float2* row2_f2 = reinterpret_cast<const float2*>(row2);
                const float2* row3_f2 = reinterpret_cast<const float2*>(row3);

                float2 pair0 = __ldg(row0_f2);
                float2 pair1 = __ldg(row1_f2);
                float2 pair2 = __ldg(row2_f2);
                float2 pair3 = __ldg(row3_f2);

                v00 = pair0.x * scale;
                v01 = pair0.y * scale;
                v10 = pair1.x * scale;
                v11 = pair1.y * scale;
                v20 = pair2.x * scale;
                v21 = pair2.y * scale;
                v30 = pair3.x * scale;
                v31 = pair3.y * scale;
            } else {
                v00 = __ldg(row0) * scale;
                v01 = __ldg(row0 + 1) * scale;
                v10 = __ldg(row1) * scale;
                v11 = __ldg(row1 + 1) * scale;
                v20 = __ldg(row2) * scale;
                v21 = __ldg(row2 + 1) * scale;
                v30 = __ldg(row3) * scale;
                v31 = __ldg(row3 + 1) * scale;
            }

            float max_val = fmaxf(fmaxf(fmaxf(v00, v01), fmaxf(v10, v11)),
                                  fmaxf(fmaxf(v20, v21), fmaxf(v30, v31)));
            thread_sum += max_val;
        }
    } else {
        for (int pool_idx = tid; pool_idx < total_pooled; pool_idx += block_size) {
            int dp = pool_idx / HpWp;
            int hp = (pool_idx - dp * HpWp) / Wp;
            int wp = pool_idx - dp * HpWp - hp * Wp;

            int d_start = dp * pool_k;
            int h_start = hp * pool_k;
            int w_start = wp * pool_k;

            float max_val = -FLT_MAX;

            for (int kd = 0; kd < pool_k; kd++) {
                int d = d_start + kd;
                if (d >= D) continue;
                for (int kh = 0; kh < pool_k; kh++) {
                    int h = h_start + kh;
                    if (h >= H) continue;
                    for (int kw = 0; kw < pool_k; kw++) {
                        int w = w_start + kw;
                        if (w >= W) continue;
                        float val = __ldg(&x_nc[d * H * W + h * W + w]) * scale;
                        max_val = fmaxf(max_val, val);
                    }
                }
            }

            thread_sum += max_val;
        }
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = block_size / 32;

    extern __shared__ float warp_sums[];
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane_id == 0) {
            float mean = val / (float)total_pooled;
            float clamped = fminf(fmaxf(mean, clamp_min), clamp_max);
            out[n * C + c] = clamped;
        }
    }
}

torch::Tensor fused_scale_maxpool_globalavgpool_clamp_cuda(torch::Tensor x, double scale, double clamp_min, double clamp_max, int64_t pool_k) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (NCDHW)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t D = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);

    int64_t Dp = (D - pool_k) / pool_k + 1;
    int64_t Hp = (H - pool_k) / pool_k + 1;
    int64_t Wp = (W - pool_k) / pool_k + 1;

    int64_t total_pooled = Dp * Hp * Wp;

    int threads = 32;
    while (threads < (int)total_pooled && threads < 256) threads <<= 1;
    if (threads < 32) threads = 32;

    int num_warps = threads / 32;
    int smem_bytes = num_warps * (int)sizeof(float);

    auto out = torch::empty({N, C, 1, 1, 1}, x.options());
    int blocks = (int)(N * C);

    fused_scale_maxpool_globalavgpool_clamp_kernel<<<blocks, threads, smem_bytes>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        (int)N, (int)C, (int)D, (int)H, (int)W,
        (int)pool_k,
        (int)Dp, (int)Hp, (int)Wp,
        static_cast<float>(scale),
        static_cast<float>(clamp_min),
        static_cast<float>(clamp_max)
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
        global average pooling, and clamps the output.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        self._cudnn_warmup_done = False
        torch.backends.cudnn.benchmark = True
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        if not self._cudnn_warmup_done:
            with torch.no_grad():
                _ = self.conv_transpose(x)
            self._cudnn_warmup_done = True
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        # scale is fused into the CUDA tail kernel
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().fused_scale_maxpool_globalavgpool_clamp_cuda(
                    x.contiguous(),
                    float(self.scale),
                    float(self.clamp_min),
                    float(self.clamp_max),
                    int(self.maxpool.kernel_size if isinstance(self.maxpool.kernel_size, int) else self.maxpool.kernel_size[0])
                )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # fused in forward_stmt_3
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # fused in forward_stmt_3
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
