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
    return f'stark_cuda_l3_p42_{digest}'

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

torch::Tensor grubidir_hidden_only(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> flat_weights,
    bool has_bias,
    bool batch_first,
    int64_t num_layers,
    int64_t hidden_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grubidir_hidden_only", &grubidir_hidden_only,
          "Bidirectional GRU hidden-only forward (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

__global__ void gru_cell_kernel(
    const float* __restrict__ gate_ih,
    const float* __restrict__ gate_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ b_hh,
    float* __restrict__ h,
    int batch,
    int H,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * H) return;
    int b = idx / H;
    int hh = idx % H;

    float r_ih = gate_ih[b * 3*H + hh];
    float z_ih = gate_ih[b * 3*H + H + hh];
    float n_ih = gate_ih[b * 3*H + 2*H + hh];

    float r_hh = gate_hh[b * 3*H + hh];
    float z_hh = gate_hh[b * 3*H + H + hh];
    float n_hh = gate_hh[b * 3*H + 2*H + hh];

    if (has_bias) {
        r_ih += b_ih[hh];
        z_ih += b_ih[H + hh];
        n_ih += b_ih[2*H + hh];
        r_hh += b_hh[hh];
        z_hh += b_hh[H + hh];
        n_hh += b_hh[2*H + hh];
    }

    float r = 1.0f / (1.0f + expf(-(r_ih + r_hh)));
    float z = 1.0f / (1.0f + expf(-(z_ih + z_hh)));
    float n = tanhf(n_ih + r * n_hh);

    float h_cur = h[b * H + hh];
    h[b * H + hh] = (1.0f - z) * n + z * h_cur;
}

__global__ void scatter_h_to_layer_input(
    const float* __restrict__ h,
    float* __restrict__ layer_in,
    int64_t t,
    int batch,
    int H,
    int dir
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * H) return;
    int b = idx / H;
    int hh = idx % H;
    layer_in[t * batch * 2 * H + b * 2 * H + dir * H + hh] = h[b * H + hh];
}

static cublasHandle_t g_cublas_handle = nullptr;

static void ensure_cublas() {
    if (g_cublas_handle == nullptr) {
        cublasCreate(&g_cublas_handle);
    }
}

torch::Tensor grubidir_hidden_only(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> flat_weights,
    bool has_bias,
    bool batch_first,
    int64_t num_layers,
    int64_t hidden_size
) {
    ensure_cublas();

    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32");
    TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32, "h0 must be CUDA float32");

    x = x.contiguous();
    h0 = h0.contiguous();

    int64_t seq_len, batch, input_size_val;
    if (batch_first) {
        batch = x.size(0);
        seq_len = x.size(1);
        input_size_val = x.size(2);
        x = x.permute({1, 0, 2}).contiguous();
    } else {
        seq_len = x.size(0);
        batch = x.size(1);
        input_size_val = x.size(2);
    }

    int64_t H = hidden_size;
    int64_t num_directions = 2;
    int weights_per_layer = has_bias ? 4 : 2;

    auto h_n = torch::empty({num_layers * num_directions, batch, H}, x.options());
    h_n.copy_(h0);

    auto gate_hh = torch::empty({batch, 3 * H}, x.options());

    auto layer_input_0 = x;
    auto layer_input_k = torch::empty({seq_len, batch, 2 * H}, x.options());

    for (int64_t layer = 0; layer < num_layers; layer++) {
        torch::Tensor cur_input;
        int64_t cur_input_size;
        if (layer == 0) {
            cur_input = layer_input_0;
            cur_input_size = input_size_val;
        } else {
            cur_input = layer_input_k;
            cur_input_size = 2 * H;
        }

        bool need_layer_out = (layer < num_layers - 1);
        float* layer_in_k_ptr = layer_input_k.data_ptr<float>();

        for (int64_t dir = 0; dir < num_directions; dir++) {
            int64_t layer_dir_idx = layer * num_directions + dir;
            int64_t w_base = layer_dir_idx * weights_per_layer;

            auto w_ih = flat_weights[w_base + 0].contiguous();
            auto w_hh = flat_weights[w_base + 1].contiguous();

            const float* b_ih_ptr = nullptr;
            const float* b_hh_ptr = nullptr;
            torch::Tensor b_ih_t, b_hh_t;
            if (has_bias) {
                b_ih_t = flat_weights[w_base + 2].contiguous();
                b_hh_t = flat_weights[w_base + 3].contiguous();
                b_ih_ptr = b_ih_t.data_ptr<float>();
                b_hh_ptr = b_hh_t.data_ptr<float>();
            }

            float* h_ptr = h_n.data_ptr<float>() + layer_dir_idx * batch * H;
            const float* wih_ptr = w_ih.data_ptr<float>();
            const float* whh_ptr = w_hh.data_ptr<float>();
            float* ghh_ptr = gate_hh.data_ptr<float>();

            auto gate_ih_seq = torch::empty({seq_len, batch, 3 * H}, x.options());
            float* gate_ih_seq_ptr = gate_ih_seq.data_ptr<float>();

            float one = 1.0f, zero = 0.0f;
            auto cur_input_flat = cur_input.reshape({seq_len * batch, cur_input_size}).contiguous();
            const float* cur_input_flat_ptr = cur_input_flat.data_ptr<float>();

            cublasSgemm(g_cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                3*H, seq_len * batch, cur_input_size,
                &one,
                wih_ptr, cur_input_size,
                cur_input_flat_ptr, cur_input_size,
                &zero,
                gate_ih_seq_ptr, 3*H);

            int threads = 256;
            int blocks_cell = (batch * H + threads - 1) / threads;

            int64_t t_start = (dir == 0) ? 0 : seq_len - 1;
            int64_t t_end   = (dir == 0) ? seq_len : -1;
            int64_t t_step  = (dir == 0) ? 1 : -1;

            for (int64_t t = t_start; t != t_end; t += t_step) {
                const float* gih_ptr_t = gate_ih_seq_ptr + t * batch * 3 * H;

                cublasSgemm(g_cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    3*H, batch, H,
                    &one,
                    whh_ptr, H,
                    h_ptr, H,
                    &zero,
                    ghh_ptr, 3*H);

                gru_cell_kernel<<<blocks_cell, threads>>>(
                    gih_ptr_t, ghh_ptr, b_ih_ptr, b_hh_ptr, h_ptr, batch, H, has_bias);

                if (need_layer_out) {
                    scatter_h_to_layer_input<<<blocks_cell, threads>>>(
                        h_ptr, layer_in_k_ptr, t, (int)batch, (int)H, (int)dir);
                }
            }
        }
    }

    return h_n;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                :param input_size: The number of expected features in the input x
                :param hidden_size: The number of features in the hidden state h
                :param num_layers: Number of recurrent layers (default: 1)
                :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
                :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
                """
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        # <<<END_IMPROVE>>>

    def forward(self, x,h0):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
                :param h_0: The initial hidden state for the input sequence, shape (num_layers * num_directions, batch_size, hidden_size) (default: None)
                :return: output, h_n
                    - output: The output features (h_t) from the last layer of the GRU, for each t, shape (seq_len, batch_size, num_directions * hidden_size) if batch_first=False, otherwise (batch_size, seq_len, num_directions * hidden_size)
                    - h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        output, h_n = self.gru(x, h0)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        return h_n
        # <<<END_IMPROVE>>>
