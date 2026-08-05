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
    return f'stark_cuda_l2_p9_{digest}'

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

torch::Tensor fused_epilogue(torch::Tensor input, float subtract_value, float multiply_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_epilogue", &fused_epilogue, "Fused subtract-multiply-ReLU epilogue");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_epilogue_kernel_vec8(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float subtract_value,
    const float multiply_value,
    const int64_t vec_size
) {
    int64_t base = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (base < vec_size) {
        float4 in0 = input[base];
        float4 out0;
        out0.x = fmaxf((in0.x - subtract_value) * multiply_value, 0.0f);
        out0.y = fmaxf((in0.y - subtract_value) * multiply_value, 0.0f);
        out0.z = fmaxf((in0.z - subtract_value) * multiply_value, 0.0f);
        out0.w = fmaxf((in0.w - subtract_value) * multiply_value, 0.0f);
        output[base] = out0;
        if (base + 1 < vec_size) {
            float4 in1 = input[base + 1];
            float4 out1;
            out1.x = fmaxf((in1.x - subtract_value) * multiply_value, 0.0f);
            out1.y = fmaxf((in1.y - subtract_value) * multiply_value, 0.0f);
            out1.z = fmaxf((in1.z - subtract_value) * multiply_value, 0.0f);
            out1.w = fmaxf((in1.w - subtract_value) * multiply_value, 0.0f);
            output[base + 1] = out1;
        }
    }
}

__global__ void fused_epilogue_kernel_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float subtract_value,
    const float multiply_value,
    const int64_t start,
    const int64_t size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < size) {
        float val = (input[idx] - subtract_value) * multiply_value;
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor fused_epilogue(torch::Tensor input, float subtract_value, float multiply_value) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int64_t size = input.numel();

    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    bool aligned = ((uintptr_t)in_ptr % 16 == 0) && ((uintptr_t)out_ptr % 16 == 0);
    int64_t vec_size = size / 4;
    int64_t tail_start = vec_size * 4;

    if (aligned && vec_size > 0) {
        // Each thread processes 2 float4 elements (8 floats total)
        int64_t threads_needed = (vec_size + 1) / 2;
        const int threads = 256;
        const int blocks = (int)((threads_needed + threads - 1) / threads);
        fused_epilogue_kernel_vec8<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(in_ptr),
            reinterpret_cast<float4*>(out_ptr),
            subtract_value,
            multiply_value,
            vec_size
        );
    } else {
        tail_start = 0;
    }

    int64_t tail_count = size - tail_start;
    if (tail_count > 0) {
        const int threads = 256;
        const int blocks = (int)((tail_count + threads - 1) / threads);
        fused_epilogue_kernel_scalar<<<blocks, threads>>>(
            in_ptr,
            out_ptr,
            subtract_value,
            multiply_value,
            tail_start,
            size
        );
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
        """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        x = self.linear(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = _stark_get_extension().fused_epilogue(x, self.subtract_value, self.multiply_value)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
