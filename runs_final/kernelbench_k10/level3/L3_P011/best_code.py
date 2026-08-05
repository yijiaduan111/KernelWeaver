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
    return f'stark_cuda_l3_p11_{digest}'

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

torch::Tensor vgg16_classifier_tail_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, torch::Tensor w3, torch::Tensor b3);

torch::Tensor vgg16_classifier_tail(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, torch::Tensor w3, torch::Tensor b3) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w1.is_cuda() && w1.dtype() == torch::kFloat32, "w1 must be CUDA float32");
    TORCH_CHECK(b1.is_cuda() && b1.dtype() == torch::kFloat32, "b1 must be CUDA float32");
    TORCH_CHECK(w2.is_cuda() && w2.dtype() == torch::kFloat32, "w2 must be CUDA float32");
    TORCH_CHECK(b2.is_cuda() && b2.dtype() == torch::kFloat32, "b2 must be CUDA float32");
    TORCH_CHECK(w3.is_cuda() && w3.dtype() == torch::kFloat32, "w3 must be CUDA float32");
    TORCH_CHECK(b3.is_cuda() && b3.dtype() == torch::kFloat32, "b3 must be CUDA float32");
    return vgg16_classifier_tail_cuda(x, w1, b1, w2, b2, w3, b3);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vgg16_classifier_tail", &vgg16_classifier_tail, "VGG16 classifier tail (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

torch::Tensor vgg16_classifier_tail_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, torch::Tensor w3, torch::Tensor b3) {
    torch::Tensor flat;
    if (x.dim() == 4) {
        int64_t batch = x.size(0);
        int64_t numel = x.size(1) * x.size(2) * x.size(3);
        flat = x.view({batch, numel}).contiguous();
    } else {
        TORCH_CHECK(x.dim() == 2, "x must be 2D or 4D");
        flat = x.contiguous();
    }

    TORCH_CHECK(flat.size(1) == w1.size(1), "flat dim mismatch w1");

    torch::Tensor h1 = at::addmm(b1, flat, w1.t());
    h1.relu_();

    torch::Tensor h2 = at::addmm(b2, h1, w2.t());
    h2.relu_();

    torch::Tensor out = at::addmm(b3, h2, w3.t());

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.features = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Block 3
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Block 4
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Block 5
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.0),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.0),
        nn.Linear(4096, num_classes)
        )
        self.to(memory_format=torch.channels_last)
        self._graph_input_shape = (10, 3, 224, 224)
        self._graph_static_input = None
        self._graph_static_output = None
        self._graph = None
        self._graph_device_index = None
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        if x.is_cuda and x.dtype == torch.float32:
            x = x.contiguous(memory_format=torch.channels_last)
        use_graph = (not self.training and x.is_cuda and x.dtype == torch.float32 and tuple(x.shape) == self._graph_input_shape)
        if use_graph:
            if self._graph is None or self._graph_device_index != x.device.index:
                self._graph_device_index = x.device.index
                self._graph_static_input = torch.empty_like(x, memory_format=torch.channels_last)
                # warmup runs
                for _ in range(3):
                    self._graph_static_input.copy_(x)
                    _y = self.features(self._graph_static_input)
                    _y = torch.flatten(_y, 1)
                    _out = self.classifier(_y)
                torch.cuda.synchronize()
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    _gy = self.features(self._graph_static_input)
                    _gy = torch.flatten(_gy, 1)
                    self._graph_static_output = self.classifier(_gy)
            self._graph_static_input.copy_(x)
            self._graph.replay()
            return self._graph_static_output.clone()
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        x = self.features(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        use_tail_ext = (not self.training and x.is_cuda and x.dtype == torch.float32 and tuple(x.shape) == (10, 512, 7, 7))
        if not use_tail_ext:
            x = torch.flatten(x, 1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        use_tail_ext = (not self.training and x.is_cuda and x.dtype == torch.float32 and (x.dim() == 4 and tuple(x.shape) == (10, 512, 7, 7) or x.dim() == 2))
        if use_tail_ext:
            ext = _stark_get_extension()
            x = ext.vgg16_classifier_tail(x, self.classifier[0].weight, self.classifier[0].bias, self.classifier[3].weight, self.classifier[3].bias, self.classifier[6].weight, self.classifier[6].bias)
        else:
            x = self.classifier(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        return x
        # <<<END_IMPROVE>>>
