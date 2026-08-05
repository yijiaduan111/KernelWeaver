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
    return f'stark_cuda_l2_p3_{digest}'

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

torch::Tensor fused_add_layernorm(torch::Tensor x, torch::Tensor sum_weight,
                                   torch::Tensor ln_weight, torch::Tensor ln_bias,
                                   double eps);

torch::Tensor fused_avgpool_gelu(torch::Tensor x,
                                  int64_t pool_kd, int64_t pool_kh, int64_t pool_kw);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_layernorm", &fused_add_layernorm,
          "Fused add + LayerNorm (CUDA)");
    m.def("fused_avgpool_gelu", &fused_avgpool_gelu,
          "Fused AvgPool3d + GELU (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Kernel 1: fused scalar-add + LayerNorm
// Input shape: (N, C, D, H, W)  --  LayerNorm([W]) normalizes over last dim
// Each thread block handles one (n,c,d,h) row of length W.
// ---------------------------------------------------------------------------
__global__ void add_layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ ln_weight,
    const float* __restrict__ ln_bias,
    float sum_weight_val,
    float eps,
    float* __restrict__ output,
    int W)
{
    // blockIdx.x = linear index over (N*C*D*H)
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_in  = input  + row * W;
    float*       row_out = output + row * W;

    // Use shared memory for the row data after add
    extern __shared__ float smem[];  // size W

    // Load and add scalar weight
    float val = 0.f;
    if (tid < W) {
        val = row_in[tid] + sum_weight_val;
        smem[tid] = val;
    }
    __syncthreads();

    // Compute mean using parallel reduction
    // We'll do two-pass in shared memory
    // Pass 1: sum
    __shared__ float s_mean;
    __shared__ float s_var;

    float thread_sum = 0.f;
    for (int i = tid; i < W; i += blockDim.x) {
        thread_sum += smem[i];
    }
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    // Only first thread in each warp has partial sum; accumulate via shared mem
    __shared__ float warp_sums[32];
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) warp_sums[warp_id] = thread_sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++) total += warp_sums[i];
        s_mean = total / (float)W;
    }
    __syncthreads();

    float mean = s_mean;

    // Pass 2: variance
    float thread_var = 0.f;
    for (int i = tid; i < W; i += blockDim.x) {
        float diff = smem[i] - mean;
        thread_var += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
    if (lane_id == 0) warp_sums[warp_id] = thread_var;
    __syncthreads();

    if (tid == 0) {
        float total = 0.f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++) total += warp_sums[i];
        s_var = total / (float)W;
    }
    __syncthreads();

    float inv_std = rsqrtf(s_var + eps);

    // Write normalized + affine output
    if (tid < W) {
        float norm_val = (smem[tid] - s_mean) * inv_std;
        row_out[tid] = norm_val * ln_weight[tid] + ln_bias[tid];
    }
}

torch::Tensor fused_add_layernorm(
    torch::Tensor x,
    torch::Tensor sum_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto sizes = x.sizes();
    // Shape: (N, C, D, H, W) -- last dim W is normalized
    int64_t W = sizes[sizes.size()-1];
    int64_t rows = x.numel() / W;

    auto output = torch::empty_like(x);

    float sw = sum_weight.item<float>();
    float ep = static_cast<float>(eps);

    // One block per row, up to 128 threads (W=64 fits in 64 threads but use 64)
    int block_size = 64;  // W=64
    if (W > 64) block_size = 128;
    // shared memory: W floats
    size_t smem_bytes = W * sizeof(float);

    add_layernorm_kernel<<<rows, block_size, smem_bytes>>>(
        x.data_ptr<float>(),
        ln_weight.data_ptr<float>(),
        ln_bias.data_ptr<float>(),
        sw, ep,
        output.data_ptr<float>(),
        static_cast<int>(W)
    );

    return output;
}

// ---------------------------------------------------------------------------
// Kernel 2: fused AvgPool3d + GELU
// Input shape: (N, C, D, H, W)
// Pool: kernel (kD,kH,kW), stride=(kD,kH,kW), padding=0
// Output shape: (N, C, D/kD, H/kH, W/kW)
// GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
// ---------------------------------------------------------------------------
__global__ void avgpool_gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int kD, int kH, int kW,
    int oD, int oH, int oW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * oD * oH * oW;
    if (idx >= total) return;

    // Decode output index
    int tmp = idx;
    int ow = tmp % oW; tmp /= oW;
    int oh = tmp % oH; tmp /= oH;
    int od = tmp % oD; tmp /= oD;
    int c  = tmp % C;  tmp /= C;
    int n  = tmp;

    // Input start
    int d_start = od * kD;
    int h_start = oh * kH;
    int w_start = ow * kW;

    float sum = 0.f;
    int pool_vol = kD * kH * kW;

    for (int kd = 0; kd < kD; kd++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int id = d_start + kd;
                int ih = h_start + kh;
                int iw = w_start + kw;
                int in_idx = ((n * C + c) * D + id) * H * W + ih * W + iw;
                sum += input[in_idx];
            }
        }
    }

    float avg = sum / (float)pool_vol;
    // erf-based GELU: 0.5 * avg * (1 + erff(avg * 0.7071067811865476f))
    float gelu_val = 0.5f * avg * (1.f + erff(avg * 0.7071067811865476f));
    output[idx] = gelu_val;
}

torch::Tensor fused_avgpool_gelu(
    torch::Tensor x,
    int64_t pool_kd, int64_t pool_kh, int64_t pool_kw)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto sizes = x.sizes();
    int N = sizes[0], C = sizes[1], D = sizes[2], H = sizes[3], W = sizes[4];
    int oD = D / pool_kd;
    int oH = H / pool_kh;
    int oW = W / pool_kw;

    auto output = torch::empty({N, C, oD, oH, oW}, x.options());

    int total = N * C * oD * oH * oW;
    int block = 256;
    int grid = (total + block - 1) / block;

    avgpool_gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        (int)pool_kd, (int)pool_kh, (int)pool_kw,
        oD, oH, oW
    );

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        x = self.conv_transpose(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = x + self.sum_weight
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.norm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = self.avg_pool(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = self.gelu(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
