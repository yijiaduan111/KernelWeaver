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
    return f'stark_cuda_l3_p39_{digest}'

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

std::vector<torch::Tensor> gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> weight_ih_list,
    std::vector<torch::Tensor> weight_hh_list,
    std::vector<torch::Tensor> bias_ih_list,
    std::vector<torch::Tensor> bias_hh_list
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gru_forward", &gru_forward_cuda, "Custom GRU forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

static cublasHandle_t g_cublas_handle = nullptr;

static void ensure_cublas() {
    if (g_cublas_handle == nullptr) {
        cublasCreate(&g_cublas_handle);
        // Use the current PyTorch CUDA stream
        cublasSetStream(g_cublas_handle, at::cuda::getCurrentCUDAStream());
    }
}

// Fused GRU cell kernel: reads precomputed input gates (bias_ih already folded in)
// and per-timestep recurrent gates (bias_hh already folded in via addmm).
__global__ void gru_cell_kernel_fused(
    const float* __restrict__ gates_ih_t,  // [batch_size, 3*hidden_size], bias_ih already added
    const float* __restrict__ gates_hh,    // [batch_size, 3*hidden_size], bias_hh already added
    const float* __restrict__ h_prev,
    float* __restrict__ h_out,
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    int b = idx / hidden_size;
    int h = idx % hidden_size;

    int base = b * 3 * hidden_size;

    float r = 1.0f / (1.0f + __expf(-(gates_ih_t[base + h] + gates_hh[base + h])));
    float z = 1.0f / (1.0f + __expf(-(gates_ih_t[base + hidden_size + h] + gates_hh[base + hidden_size + h])));
    float n = tanhf(gates_ih_t[base + 2*hidden_size + h] + r * gates_hh[base + 2*hidden_size + h]);

    float h_prev_val = h_prev[b * hidden_size + h];
    h_out[b * hidden_size + h] = (1.0f - z) * n + z * h_prev_val;
}

std::vector<torch::Tensor> gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> weight_ih_list,
    std::vector<torch::Tensor> weight_hh_list,
    std::vector<torch::Tensor> bias_ih_list,
    std::vector<torch::Tensor> bias_hh_list
) {
    ensure_cublas();
    // Keep cuBLAS on the current stream in case it changed
    cublasSetStream(g_cublas_handle, at::cuda::getCurrentCUDAStream());

    int seq_len    = x.size(0);
    int batch_size = x.size(1);
    int num_layers = (int)weight_ih_list.size();
    int hidden_size = (int)weight_hh_list[0].size(1);
    int gate_size   = 3 * hidden_size;

    // x and h0 are already contiguous from Python side
    auto h_n = torch::empty({num_layers, batch_size, hidden_size}, x.options());
    auto gates_hh = torch::empty({batch_size, gate_size}, x.options());

    const float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    torch::Tensor layer_input = x;

    for (int layer = 0; layer < num_layers; layer++) {
        auto W_ih = weight_ih_list[layer].contiguous();
        auto W_hh = weight_hh_list[layer].contiguous();
        auto b_ih = bias_ih_list[layer].contiguous();
        auto b_hh = bias_hh_list[layer].contiguous();

        int in_feat = (int)W_ih.size(1);

        // Precompute all input projections at once: [seq_len*batch, gate_size]
        // Use addmm-style: result = b_ih (broadcast) + layer_input_flat @ W_ih^T
        // at::addmm expects (bias[gate_size], mat1[M,K], mat2[K,N])
        // layer_input: [seq_len, batch, in_feat] -> view [seq_len*batch, in_feat]
        auto layer_flat = layer_input.reshape({seq_len * batch_size, in_feat});
        // gates_ih_all: [seq_len*batch, gate_size] with bias_ih folded in
        auto gates_ih_all = at::addmm(b_ih, layer_flat, W_ih.t());
        // Shape: [seq_len, batch_size, gate_size]
        gates_ih_all = gates_ih_all.view({seq_len, batch_size, gate_size}).contiguous();

        auto h_prev = h0[layer].contiguous().clone();
        auto layer_output = torch::empty({seq_len, batch_size, hidden_size}, x.options());

        int total = batch_size * hidden_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        for (int t = 0; t < seq_len; t++) {
            // Recurrent projection: gates_hh = h_prev @ W_hh^T + b_hh
            // Use cuBLAS: C = alpha * W_hh * h_prev^T + beta * C
            // gates_hh[batch, gate_size], h_prev[batch, hidden]
            cublasSgemm(
                g_cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                gate_size, batch_size, hidden_size,
                &alpha,
                W_hh.data_ptr<float>(), hidden_size,
                h_prev.data_ptr<float>(), hidden_size,
                &beta_zero,
                gates_hh.data_ptr<float>(), gate_size
            );

            // Add b_hh to gates_hh: use a simple kernel-free approach via ATen
            // gates_hh view as [batch, gate_size] and add b_hh [gate_size]
            // We do this inside the cell kernel to keep launch count low â just pass b_hh pointer
            // Actually b_hh is already a separate arg; let's fold it via a cublas call:
            // Instead, add b_hh using ATen (one broadcast add per timestep is cheap vs a GEMM)
            // We'll do it by treating gates_hh as a [batch, gate_size] tensor and calling add_
            {
                auto gh_view = gates_hh.view({batch_size, gate_size});
                gh_view.add_(b_hh);
            }

            auto h_out = layer_output[t];
            gru_cell_kernel_fused<<<blocks, threads>>>(
                gates_ih_all[t].data_ptr<float>(),
                gates_hh.data_ptr<float>(),
                h_prev.data_ptr<float>(),
                h_out.data_ptr<float>(),
                batch_size,
                hidden_size
            );

            h_prev = layer_output[t].contiguous();
        }

        h_n[layer] = h_prev;
        layer_input = layer_output;
    }

    return {layer_input.contiguous(), h_n};
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        self.gru.flatten_parameters()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bias = bias
        self._batch_first = batch_first
        # <<<END_IMPROVE>>>

    def forward(self, x,h0):
        # <<<IMPROVE:forward_stmt_1>>>
        x = x.contiguous()
        h0 = h0.contiguous()

        if self._batch_first:
            x = x.transpose(0, 1)

        use_fast_path = (
            x.is_cuda and h0.is_cuda and
            x.dtype == torch.float32 and h0.dtype == torch.float32 and
            x.is_contiguous() and h0.is_contiguous() and
            not self.gru.training
        )
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if use_fast_path:
            weight_ih_list = [getattr(self.gru, f'weight_ih_l{i}') for i in range(self._num_layers)]
            weight_hh_list = [getattr(self.gru, f'weight_hh_l{i}') for i in range(self._num_layers)]
            bias_ih_list = [getattr(self.gru, f'bias_ih_l{i}') for i in range(self._num_layers)]
            bias_hh_list = [getattr(self.gru, f'bias_hh_l{i}') for i in range(self._num_layers)]

            ext = _stark_get_extension()
            output, h_n = ext.gru_forward(x, h0, weight_ih_list, weight_hh_list, bias_ih_list, bias_hh_list)
        else:
            output, h_n = self.gru(x, h0)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        if self._batch_first:
            output = output.transpose(0, 1)

        return output
        # <<<END_IMPROVE>>>
