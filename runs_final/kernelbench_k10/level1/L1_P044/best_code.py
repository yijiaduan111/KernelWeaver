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
    return f'stark_cuda_l1_p44_{digest}'

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

torch::Tensor avg_pool1d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda, "1D Average Pooling CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_W 256

// Specialized kernel for kernel_size=8, stride=1, padding=4
// Uses 128 threads, one output per thread, compile-time unrolled window sum
// smem covers BLOCK_K8 outputs + 7 halo = 135 floats per block
#define BLOCK_K8 128

__global__ void __launch_bounds__(BLOCK_K8, 8) avg_pool1d_k8_s1_p4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int L_in,
    int L_out
) {
    int nc = blockIdx.y;
    int tile_start = blockIdx.x * BLOCK_K8;  // first output index in this tile
    int tid = threadIdx.x;

    const float* in_row = input + (long long)nc * L_in;
    float* out_row = output + (long long)nc * L_out;

    // smem covers input positions [tile_start - 4, tile_start - 4 + BLOCK_K8 + 7)
    // = BLOCK_K8 + 7 = 135 elements
    const int SMEM_N = BLOCK_K8 + 7;
    extern __shared__ float smem[];

    int tile_in_start = tile_start - 4;  // padding=4

    // Collaboratively load SMEM_N elements (135 elements, 128 threads: 2 passes)
    for (int i = tid; i < SMEM_N; i += BLOCK_K8) {
        int in_idx = tile_in_start + i;
        smem[i] = (in_idx >= 0 && in_idx < L_in) ? in_row[in_idx] : 0.0f;
    }
    __syncthreads();

    int out_idx = tile_start + tid;
    if (out_idx < L_out) {
        // Each thread sums exactly 8 elements starting at smem[tid]
        float sum;
        sum  = smem[tid];
        sum += smem[tid + 1];
        sum += smem[tid + 2];
        sum += smem[tid + 3];
        sum += smem[tid + 4];
        sum += smem[tid + 5];
        sum += smem[tid + 6];
        sum += smem[tid + 7];
        out_row[out_idx] = sum * 0.125f;
    }
}

// Shared memory tiled kernel for general cases
__global__ void avg_pool1d_smem_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding
) {
    int nc = blockIdx.y;
    int tile_start = blockIdx.x * BLOCK_W;
    int tid = threadIdx.x;

    const float* in_row = input + (long long)nc * L_in;
    float* out_row = output + (long long)nc * L_out;

    int smem_in_start = tile_start * stride - padding;
    int smem_size = BLOCK_W * stride + kernel_size - 1;

    extern __shared__ float smem[];

    for (int i = tid; i < smem_size; i += BLOCK_W) {
        int in_idx = smem_in_start + i;
        smem[i] = (in_idx >= 0 && in_idx < L_in) ? in_row[in_idx] : 0.0f;
    }
    __syncthreads();

    int out_idx = tile_start + tid;
    if (out_idx < L_out) {
        float sum = 0.0f;
        int smem_offset = tid * stride;
        #pragma unroll 8
        for (int k = 0; k < kernel_size; k++) {
            sum += smem[smem_offset + k];
        }
        out_row[out_idx] = sum / (float)kernel_size;
    }
}

// Generic fallback kernel: one thread per output element
__global__ void avg_pool1d_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding
) {
    int nc = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= L_out) return;

    const float* in_row = input + (long long)nc * L_in;
    float* out_row = output + (long long)nc * L_out;

    int start = out_idx * stride - padding;
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int in_idx = start + k;
        if (in_idx >= 0 && in_idx < L_in) {
            sum += in_row[in_idx];
        }
    }
    out_row[out_idx] = sum / (float)kernel_size;
}

torch::Tensor avg_pool1d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (N, C, L)");

    int N = x.size(0);
    int C = x.size(1);
    int L_in = x.size(2);
    int L_out = (L_in + 2 * padding - kernel_size) / stride + 1;
    int NC = N * C;

    auto output = torch::empty({N, C, L_out}, x.options());

    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    // Specialized path for k=8, s=1, p=4
    if (kernel_size == 8 && stride == 1 && padding == 4) {
        const int SMEM_N = BLOCK_K8 + 7;  // 135
        int smem_bytes = SMEM_N * sizeof(float);
        dim3 grid((L_out + BLOCK_K8 - 1) / BLOCK_K8, NC);
        dim3 block(BLOCK_K8);
        avg_pool1d_k8_s1_p4_kernel<<<grid, block, smem_bytes>>>(
            in_ptr, out_ptr, L_in, L_out
        );
        return output;
    }

    // Use shared memory tiled kernel for other cases
    int smem_size = (BLOCK_W * (int)stride + (int)kernel_size - 1) * sizeof(float);
    if (smem_size <= 48 * 1024) {
        dim3 grid((L_out + BLOCK_W - 1) / BLOCK_W, NC);
        dim3 block(BLOCK_W);
        avg_pool1d_smem_kernel<<<grid, block, smem_size>>>(
            in_ptr, out_ptr, L_in, L_out, (int)kernel_size, (int)stride, (int)padding
        );
    } else {
        dim3 grid((L_out + BLOCK_W - 1) / BLOCK_W, NC);
        dim3 block(BLOCK_W);
        avg_pool1d_generic_kernel<<<grid, block>>>(
            in_ptr, out_ptr, L_in, L_out, (int)kernel_size, (int)stride, (int)padding
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Simple model that performs 1D Average Pooling.
        """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initializes the 1D Average Pooling layer.

                Args:
                    kernel_size (int): Size of the pooling window.
                    stride (int, optional): Stride of the pooling operation. Defaults to 1.
                    padding (int, optional): Padding applied to the input tensor. Defaults to 0.
                """
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Applies 1D Average Pooling to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

                Returns:
                    torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if x.is_cuda and x.dtype == torch.float32:
            x_c = x.contiguous()
            ks = self.avg_pool.kernel_size
            st = self.avg_pool.stride
            pd = self.avg_pool.padding
            if isinstance(ks, (list, tuple)): ks = ks[0]
            if isinstance(st, (list, tuple)): st = st[0]
            if isinstance(pd, (list, tuple)): pd = pd[0]
            return _stark_get_extension().avg_pool1d_cuda(x_c, ks, st, pd)
        return self.avg_pool(x)
        # <<<END_IMPROVE>>>
