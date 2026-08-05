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
    return f'stark_cuda_l3_p15_{digest}'

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

torch::Tensor densenet121_fused_tail_packed(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor shift
);

torch::Tensor densenet121_transition_fused(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps,
    torch::Tensor conv_weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("densenet121_fused_tail_packed", &densenet121_fused_tail_packed, "DenseNet121 fused final BN+ReLU+GAP with prepacked scale/shift (CUDA)");
    m.def("densenet121_transition_fused", &densenet121_transition_fused, "DenseNet121 fused TransitionLayer BN+ReLU+Conv1x1+AvgPool2x2 (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Fused final BN+ReLU+GAP kernel (prepacked scale/shift)
// ---------------------------------------------------------------------------
__global__ void densenet121_fused_tail_packed_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    float* __restrict__ out,
    int N, int C, int HW
) {
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;

    float sc = scale[c];
    float sh = shift[c];

    float sum = 0.0f;
    const float* row = x + ((n * C) + c) * HW;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = row[i] * sc + sh;
        sum += fmaxf(val, 0.0f);
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[nc] = smem[0] / float(HW);
    }
}

torch::Tensor densenet121_fused_tail_packed(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor shift
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(scale.is_cuda() && shift.is_cuda(), "scale and shift must be on CUDA");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;

    auto out = torch::empty({N, C}, x.options());

    densenet121_fused_tail_packed_kernel<<<N * C, 256>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        shift.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, HW
    );

    return out;
}

// ---------------------------------------------------------------------------
// Fused TransitionLayer: BN inference + ReLU + bias-free 1x1 Conv + 2x2 AvgPool
// Grid: (N, OC, OH*OW) where OH = H/2, OW = W/2
// Each thread computes one output element (n, oc, oh, ow).
// ---------------------------------------------------------------------------
__global__ void densenet121_transition_fused_kernel(
    const float* __restrict__ x,         // (N, IC, H, W)
    const float* __restrict__ bn_scale,  // (IC,) precomputed weight/sqrt(var+eps)
    const float* __restrict__ bn_shift,  // (IC,) precomputed bias - mean*scale
    const float* __restrict__ conv_w,    // (OC, IC) - 1x1 conv weight
    float* __restrict__ out,             // (N, OC, OH, OW)
    int N, int IC, int OC,
    int H, int W,
    int OH, int OW
) {
    // blockIdx.x: output spatial tile index
    // blockIdx.y: output channel OC tile
    // blockIdx.z: batch
    int ow = (blockIdx.x * blockDim.x + threadIdx.x) % OW;
    int oh = (blockIdx.x * blockDim.x + threadIdx.x) / OW;
    int oc = blockIdx.y;
    int n  = blockIdx.z;

    if (oh >= OH || ow >= OW) return;

    // The 4 source pixels in the 2x2 avg-pool window
    int ih0 = oh * 2;
    int iw0 = ow * 2;

    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;

    // Tile over input channels to reduce register pressure
    // Each iteration we load bn_scale/shift and conv_w[oc, ic]
    const float* conv_row = conv_w + oc * IC;

    for (int ic = 0; ic < IC; ic++) {
        float sc = __ldg(bn_scale + ic);
        float sh = __ldg(bn_shift + ic);
        float cw = __ldg(conv_row + ic);

        int base = (n * IC + ic) * H * W;

        float v00 = __ldg(x + base + ih0 * W + iw0)       * sc + sh;
        float v01 = __ldg(x + base + ih0 * W + iw0 + 1)   * sc + sh;
        float v10 = __ldg(x + base + (ih0+1) * W + iw0)   * sc + sh;
        float v11 = __ldg(x + base + (ih0+1) * W + iw0+1) * sc + sh;

        acc00 += fmaxf(v00, 0.f) * cw;
        acc01 += fmaxf(v01, 0.f) * cw;
        acc10 += fmaxf(v10, 0.f) * cw;
        acc11 += fmaxf(v11, 0.f) * cw;
    }

    float pool_val = (acc00 + acc01 + acc10 + acc11) * 0.25f;
    out[((n * OC + oc) * OH + oh) * OW + ow] = pool_val;
}

torch::Tensor densenet121_transition_fused(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps,
    torch::Tensor conv_weight
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(conv_weight.dim() == 4, "conv_weight must be 4D (OC,IC,1,1)");

    int N  = x.size(0);
    int IC = x.size(1);
    int H  = x.size(2);
    int W  = x.size(3);
    int OC = conv_weight.size(0);
    int OH = H / 2;
    int OW = W / 2;

    // Precompute BN inference coefficients on GPU
    auto var_eps = bn_running_var + (float)bn_eps;
    auto bn_scale = (bn_weight * var_eps.rsqrt()).contiguous();
    auto bn_shift = (bn_bias - bn_running_mean * bn_scale).contiguous();
    // conv_weight is (OC, IC, 1, 1) - squeeze to (OC, IC)
    auto conv_w2d = conv_weight.view({OC, IC}).contiguous();

    auto out = torch::empty({N, OC, OH, OW}, x.options());

    int spatial_out = OH * OW;
    // threads per block: up to 256, cover spatial_out
    int threads = 128;
    int blocks_x = (spatial_out + threads - 1) / threads;

    dim3 grid(blocks_x, OC, N);
    densenet121_transition_fused_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        bn_scale.data_ptr<float>(),
        bn_shift.data_ptr<float>(),
        conv_w2d.data_ptr<float>(),
        out.data_ptr<float>(),
        N, IC, OC, H, W, OH, OW
    );

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
        block_layers = [6, 12, 24, 16]
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
        self._final_bn_packed_scale = None
        self._final_bn_packed_shift = None
        self._final_bn_pack_key = None
        self._use_transition_fusion = True
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
                        _can_fuse = (
                            self._use_transition_fusion
                            and not self.training
                            and x.is_cuda
                            and x.dtype == torch.float32
                            and x.is_contiguous()
                            and x.dim() == 4
                        )
                        if _can_fuse:
                            try:
                                tl = self.transition_layers[i]
                                norm = tl.norm
                                conv = tl.conv
                                x = _stark_get_extension().densenet121_transition_fused(
                                    x,
                                    norm.weight,
                                    norm.bias,
                                    norm.running_mean,
                                    norm.running_var,
                                    float(norm.eps),
                                    conv.weight
                                )
                            except Exception:
                                x = self.transition_layers[i](x)
                        else:
                            x = self.transition_layers[i](x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        _use_fused = (x.is_cuda and not self.training and not self.final_bn.training
        and x.dtype == torch.float32 and x.is_contiguous() and x.dim() == 4)
        if _use_fused:
            _key = (
            x.device,
            x.dtype,
            self.final_bn.weight.data_ptr(),
            self.final_bn.bias.data_ptr(),
            self.final_bn.running_mean.data_ptr(),
            self.final_bn.running_var.data_ptr(),
            float(self.final_bn.eps),
            )
            if _key != self._final_bn_pack_key:
                _var_eps = self.final_bn.running_var + self.final_bn.eps
                _scale = (self.final_bn.weight * torch.rsqrt(_var_eps)).contiguous()
                _shift = (self.final_bn.bias - self.final_bn.running_mean * _scale).contiguous()
                self._final_bn_packed_scale = _scale
                self._final_bn_packed_shift = _shift
                self._final_bn_pack_key = _key
            x = _stark_get_extension().densenet121_fused_tail_packed(
            x,
            self._final_bn_packed_scale,
            self._final_bn_packed_shift
            )
        else:
            x = self.final_bn(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        if x.dim() == 4:
                    x = F.relu(x, inplace=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        x = self.classifier(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        return x
        # <<<END_IMPROVE>>>
