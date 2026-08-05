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
    return f'stark_cuda_l3_p3_{digest}'

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
#include <vector>

torch::Tensor deepnarrowmlp_forward(
    torch::Tensor x,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases) {

    TORCH_CHECK(weights.size() == biases.size(), "weights and biases must have the same length");
    TORCH_CHECK(x.dim() == 2, "input must be 2D");

    torch::NoGradGuard no_grad;
    torch::Tensor out = x;
    const int64_t nlayers = static_cast<int64_t>(weights.size());
    for (int64_t i = 0; i < n_layers; ++i) {
        out = torch::addmm(biases[i], out, weights[i].t());
        if (i < n_layers - 1) {
            out = torch::relu_(out);
        }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deepnarrowmlp_forward", &deepnarrowmlp_forward, "DepNarrowMLP forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Add CUDA kernels and exported wrapper functions here.
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                :param input_size: The number of input features
                :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
                :param output_size: The number of output features
                """
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
                    layers.append(nn.Linear(current_input_size, hidden_size))
                    layers.append(nn.ReLU())
                    current_input_size = hidden_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)
        self._graph = None
        self._graph_input = None
        self._graph_output = None
        self._graph_shape = None
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        native_fastpath = x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and not self.training
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if native_fastpath:
            weights = [m.weight for m in self.network if isinstance(m, nn.Linear)]
            biases = [m.bias for m in self.network if isinstance(m, nn.Linear)]
            return _stark_get_extension().deepnarrowmlp_forward(x, weights, biases)
        return self.network(x)
        # <<<END_IMPROVE>>>
