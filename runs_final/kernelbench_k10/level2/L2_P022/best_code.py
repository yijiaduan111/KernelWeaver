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
    return f'stark_cuda_l2_p22_{digest}'

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

torch::Tensor fused_post_ops_cuda(torch::Tensor x, double scale_factor, double clamp_min, double clamp_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_ops", &fused_post_ops_cuda, "Fused scale+add+clamp+logsumexp+mish (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_2D(x) TORCH_CHECK(x.dim() == 2, #x " must be 2D")

__global__ void fused_post_ops_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B,
    int H,
    float scale_combined,
    float clamp_min,
    float clamp_max
) {
    int row = blockIdx.x;
    if (row >= B) return;

    const int BLOCK_SIZE = 256;
    int tid = threadIdx.x;

    const float* row_ptr = input + row * H;
    const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
    int H4 = H >> 2;
    int limit = H4 & ~1;

    float local_max = -1e38f;
    float local_sum = 0.0f;

    // 2x unrolled vectorized loop
    for (int i = tid; i < limit; i += BLOCK_SIZE * 2) {
        // First float4
        float4 v4a = __ldg(&row4[i]);

        float v = v4a.x * scale_combined;
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }

        v = v4a.y * scale_combined;
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }

        v = v4a.z * scale_combined;
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }

        v = v4a.w * scale_combined;
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }

        // Second float4
        int j = i + BLOCK_SIZE;
        if (j < H4) {
            float4 v4b = __ldg(&row4[j]);

            v = v4b.x * scale_combined;
            v = fmaxf(clamp_min, fminf(clamp_max, v));
            if (v > local_max) {
                local_sum = local_sum * expf(local_max - v) + 1.0f;
                local_max = v;
            } else {
                local_sum += expf(v - local_max);
            }

            v = v4b.y * scale_combined;
            v = fmaxf(clamp_min, fminf(clamp_max, v));
            if (v > local_max) {
                local_sum = local_sum * expf(local_max - v) + 1.0f;
                local_max = v;
            } else {
                local_sum += expf(v - local_max);
            }

            v = v4b.z * scale_combined;
            v = fmaxf(clamp_min, fminf(clamp_max, v));
            if (v > local_max) {
                local_sum = local_sum * expf(local_max - v) + 1.0f;
                local_max = v;
            } else {
                local_sum += expf(v - local_max);
            }

            v = v4b.w * scale_combined;
            v = fmaxf(clamp_min, fminf(clamp_max, v));
            if (v > local_max) {
                local_sum = local_sum * expf(local_max - v) + 1.0f;
                local_max = v;
            } else {
                local_sum += expf(v - local_max);
            }
        }
    }

    // Cleanup loop for leftover single float4
    for (int i = tid + limit; i < H4; i += BLOCK_SIZE) {
        float4 v4 = __ldg(&row4[i]);

        float v = v4.x * scale_combined;
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }

        v = v4.y * scale_combined;
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }

        v = v4.z * scale_combined;
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }

        v = v4.w * scale_combined;
        v = fmaxf(clamp_min, fminf(clamp_max, v));
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }
    }

    // Scalar tail for H not divisible by 4
    for (int i = (H4 << 2) + tid; i < H; i += BLOCK_SIZE) {
        float vt = __ldg(&row_ptr[i]) * scale_combined;
        vt = fmaxf(clamp_min, fminf(clamp_max, vt));
        if (vt > local_max) {
            local_sum = local_sum * expf(local_max - vt) + 1.0f;
            local_max = vt;
        } else {
            local_sum += expf(vt - local_max);
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, local_sum, offset);
        if (other_max > local_max) {
            local_sum = local_sum * expf(local_max - other_max) + other_sum;
            local_max = other_max;
        } else {
            local_sum += other_sum * expf(other_max - local_max);
        }
    }

    __shared__ float warp_max[8];
    __shared__ float warp_sum[8];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float m = (lane_id < 8) ? warp_max[lane_id] : -1e38f;
        float s = (lane_id < 8) ? warp_sum[lane_id] : 0.0f;

        for (int offset = 4; offset > 0; offset >>= 1) {
            float other_m = __shfl_xor_sync(0xffffffff, m, offset);
            float other_s = __shfl_xor_sync(0xffffffff, s, offset);
            if (other_m > m) {
                s = s * expf(m - other_m) + other_s;
                m = other_m;
            } else {
                s += other_s * expf(other_m - m);
            }
        }

        if (lane_id == 0) {
            float z = m + logf(s);
            float sp;
            if (z > 20.0f) {
                sp = z;
            } else {
                sp = log1pf(expf(z));
            }
            float mish_z = z * tanhf(sp);
            output[row] = z * mish_z;
        }
    }
}

torch::Tensor fused_post_ops_cuda(torch::Tensor x, double scale_factor, double clamp_min, double clamp_max) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    CHECK_2D(x);

    int B = x.size(0);
    int H = x.size(1);

    auto output = torch::empty({B, 1}, x.options());

    float scale_combined = (float)(scale_factor * 2.0);
    float cmin = (float)clamp_min;
    float cmax = (float)clamp_max;

    dim3 grid(B);
    dim3 block(256);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    fused_post_ops_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H,
        scale_combined,
        cmin,
        cmax
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
        applies LogSumExp, and finally applies the Mish activation function.
        """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x: Input tensor of shape (batch_size, input_size).

                Returns:
                    Output tensor of shape (batch_size, hidden_size).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.matmul(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = _stark_get_extension().fused_post_ops(x, float(self.scale_factor), float(self.clamp_min), float(self.clamp_max))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        # fused in fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        # fused in fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        # fused in fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        # fused in fused_post_ops
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        return x
        # <<<END_IMPROVE>>>
