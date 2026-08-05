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
    return f'stark_cuda_l2_p28_{digest}'

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

torch::Tensor rowwise_norm_add_mul_cuda(torch::Tensor x, torch::Tensor y, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_norm_add_mul", &rowwise_norm_add_mul_cuda, "Rowwise norm + add + multiply (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rowwise_norm_add_mul_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int64_t rows,
    int64_t cols,
    float eps
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * cols;
    const float* y_row = y + row * cols;
    float* out_row = out + row * cols;

    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int64_t i = tid; i < cols; i += block_size) {
        float val = x_row[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        __syncthreads();
    }

    __shared__ float mean;
    __shared__ float inv_std;

    if (tid == 0) {
        mean = shared_sum[0] / cols;
        float variance = (shared_sq_sum[0] / cols) - (mean * mean);
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int64_t i = tid; i < cols; i += block_size) {
        float normalized = (x_row[i] - mean) * inv_std;
        float y_val = y_row[i];
        out_row[i] = (normalized + y_val) * y_val;
    }
}

torch::Tensor rowwise_norm_add_mul_cuda(torch::Tensor x, torch::Tensor y, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(y.dim() == 2, "y must be 2D");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");

    auto out = torch::empty_like(x);
    int64_t rows = x.size(0);
    int64_t cols = x.size(1);

    const int threads = 256;
    const int blocks = rows;

    rowwise_norm_add_mul_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols,
        static_cast<float>(eps)
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
        """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        # <<<END_IMPROVE>>>

    def forward(self, x, y):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_features).
                    y (torch.Tensor): Input tensor of shape (batch_size, out_features).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_features).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.bmm(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if x.is_cuda and y.is_cuda and x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float32 and y.dtype == torch.float32 and x.dim() == 2 and y.dim() == 2 and x.shape == y.shape:
            ext = _stark_get_extension()
            x = ext.rowwise_norm_add_mul(x, y, self.instance_norm.eps)
            fused_path_taken = True
        else:
            x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
            fused_path_taken = False
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if not fused_path_taken:
                    x = x + y
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if not fused_path_taken:
                    x = x * y
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        return x
        # <<<END_IMPROVE>>>
