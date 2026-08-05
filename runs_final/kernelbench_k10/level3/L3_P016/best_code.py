import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return f'stark_cuda_l3_p16_{digest}'

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

torch::Tensor fused_final_bn_relu_gap(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_final_bn_relu_gap", &fused_final_bn_relu_gap,
          "Fused BatchNorm2d + ReLU + GlobalAveragePool (eval-only)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Each thread owns one (n, c) lane and iterates over the full H*W spatial extent.
// Normalises with running stats, applies affine + ReLU in-register, then
// reduces to a scalar average -- no intermediate (N,C,H,W) tensor is written.
template <typename scalar_t>
__global__ void bn_relu_gap_kernel(
    const scalar_t* __restrict__ x,
    const float*__restrict__ running_mean,
    const float*    __restrict__ running_var,
    const float*    __restrict__ weight,
    const float*    __restrict__ bias_bn,
    scalar_t*__restrict__ out,
    int64_t N, int64_t C, int64_t H, int64_t W,
    float eps)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = N * C;
    if (idx >= total) return;

    const int64_t n = idx / C;
    const int64_t c = idx % C;
    const int64_t spatial = H * W;

    const float mean    = running_mean[c];
    const float var     = running_var[c];
    const float gamma   = weight[c];
    const float beta    = bias_bn[c];
    const float inv_std = rsqrtf(var + eps);
    const float scale   = gamma * inv_std;// precomputed per-channel
    const float shift   = beta - mean * scale;

    const scalar_t* x_nc = x + (n * C + c) * spatial;
    float sum = 0.0f;

    for (int64_t s = 0; s < spatial; ++s) {
        float v = static_cast<float>(x_nc[s]) * scale + shift;
        if (v > 0.0f) sum += v;   // ReLU fused into accumulation
    }

    out[idx] = static_cast<scalar_t>(sum / static_cast<float>(spatial));
}

torch::Tensor fused_final_bn_relu_gap(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps)
{
    TORCH_CHECK(x.dim() == 4, "fused_final_bn_relu_gap: x must be 4-D NCHW");
    TORCH_CHECK(x.is_cuda(), "fused_final_bn_relu_gap: x must be a CUDA tensor");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    // Output is already-flattened (N, C) -- matches what the classifier expects.
    auto out = torch::empty({N, C}, x.options());

    // BN parameters are always stored as float32 in PyTorch; .contiguous() is
    // typically a no-op here but ensures pointer validity.
    auto rm = running_mean.contiguous();
    auto rv = running_var.contiguous();
    auto w  = weight.contiguous();
    auto b  = bias.contiguous();

    const int64_t total= N * C;
    const int     block_size = 256;
    const int64_t grid_size  = (total + block_size - 1) / block_size;

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "bn_relu_gap_kernel", [&]() {
        bn_relu_gap_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            rm.data_ptr<float>(),
            rv.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            N, C, H, W,
            static_cast<float>(eps)
        );
    });

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                :param growth_rate: The growth rate of the DenseNet (new features added per layer)
                :param num_classes: The number of output classes for classification
                """
        self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
        num_features = 64
        block_layers = [6, 12, 48, 32]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
                    block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
                    self.dense_blocks.append(block)
                    num_features = num_features + num_layers * growth_rate

                    if i != len(block_layers) - 1:
                        transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                        self.transition_layers.append(transition)
                        num_features = num_features // 2
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                :param x: Input tensor of shape (batch_size, 3, height, width)
                :return: Output tensor of shape (batch_size, num_classes)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.features(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        for i, block in enumerate(self.dense_blocks):
                    x = block(x)
                    if i != len(self.dense_blocks) - 1:
                        x = self.transition_layers[i](x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if (not self.training
                and x.is_cuda
                and x.is_contiguous()
                and x.is_floating_point()
                and x.dim() == 4
                and self.final_bn.running_mean is not None
                and self.final_bn.running_var is not None
                and self.final_bn.weight is not None
                and self.final_bn.bias is not None):
            x = _stark_get_extension().fused_final_bn_relu_gap(
                x,
                self.final_bn.running_mean,
                self.final_bn.running_var,
                self.final_bn.weight,
                self.final_bn.bias,
                self.final_bn.eps,
            )
        else:
            x = self.final_bn(x)
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        x = self.classifier(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        return x
        # <<<END_IMPROVE>>>
