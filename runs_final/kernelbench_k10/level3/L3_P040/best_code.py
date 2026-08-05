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
    return f'stark_cuda_l3_p40_{digest}'

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

// Forward declaration of CUDA implementation
torch::Tensor gruhidden_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> w_ih,
    std::vector<torch::Tensor> w_hh,
    std::vector<torch::Tensor> b_ih,
    std::vector<torch::Tensor> b_hh,
    bool batch_first
);

torch::Tensor gruhidden_forward(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> w_ih,
    std::vector<torch::Tensor> w_hh,
    std::vector<torch::Tensor> b_ih,
    std::vector<torch::Tensor> b_hh,
    bool batch_first
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(h0.is_cuda(), "h0 must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(h0.dtype() == torch::kFloat32, "h0 must be float32");
    return gruhidden_forward_cuda(
        x.contiguous(), h0.contiguous(),
        w_ih, w_hh, b_ih, b_hh, batch_first
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gruhidden_forward", &gruhidden_forward,
          "GRU forward computing only h_n (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// GRU gate kernel: [r, z, n] gate order, in-place hidden update.
// pre_ih: (batch, 3*H), pre_hh: (batch, 3*H), h: (batch, H)
__global__ void gru_cell_kernel(
    const float* __restrict__ pre_ih,
    const float* __restrict__ pre_hh,
    float* __restrict__ h,
    int batch_size,
    int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * hidden_size;
    if (tid >= total) return;

    int b = tid / hidden_size;
    int j = tid % hidden_size;
    int H = hidden_size;

    float r_ih = pre_ih[b * 3*H + j];
    float z_ih = pre_ih[b * 3*H + H + j];
    float n_ih = pre_ih[b * 3*H + 2*H + j];

    float r_hh = pre_hh[b * 3*H + j];
    float z_hh = pre_hh[b * 3*H + H + j];
    float n_hh = pre_hh[b * 3*H + 2*H + j];

    float r = 1.0f / (1.0f + __expf(-(r_ih + r_hh)));
    float z = 1.0f / (1.0f + __expf(-(z_ih + z_hh)));
    float n = tanhf(n_ih + r * n_hh);

    float h_prev = h[b * H + j];
    h[b * H + j] = (1.0f - z) * n + z * h_prev;
}

// Fused bias-add + copy kernel for inter-layer handoff.
// Adds bias to a preactivation buffer row-wise.
__global__ void add_bias_kernel(
    float* __restrict__ mat,
    const float* __restrict__ bias,
    int total,
    int cols
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    mat[tid] += bias[tid % cols];
}

// Same-stream device-to-device copy kernel to replace cudaMemcpyAsync(stream=0).
__global__ void d2d_copy_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = src[tid];
}

// Per-device cuBLAS handle cache.
static cublasHandle_t cublas_handles[64] = {};
static bool cublas_handles_init[64] = {};

static cublasHandle_t get_cublas_handle(int device_id) {
    if (!cublas_handles_init[device_id]) {
        cublasCreate(&cublas_handles[device_id]);
        cublas_handles_init[device_id] = true;
    }
    return cublas_handles[device_id];
}

// Row-major GEMM: C(M,N) = A(M,K) * B^T(N,K)
static void sgemm_NT(
    cublasHandle_t handle,
    int M, int N, int K,
    const float* A, const float* B, float* C,
    float alpha = 1.0f, float beta = 0.0f
) {
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, K,
                A, K,
                &beta,
                C, N);
}

torch::Tensor gruhidden_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> w_ih,
    std::vector<torch::Tensor> w_hh,
    std::vector<torch::Tensor> b_ih,
    std::vector<torch::Tensor> b_hh,
    bool batch_first
) {
    // Normalize layout to (seq_len, batch, input_size) once.
    torch::Tensor x_seq;
    if (batch_first) {
        x_seq = x.permute({1, 0, 2}).contiguous();
    } else {
        x_seq = x;  // already contiguous from caller
    }

    int seq_len    = (int)x_seq.size(0);
    int batch_size = (int)x_seq.size(1);
    int input_size = (int)x_seq.size(2);
    int num_layers  = (int)w_ih.size();
    int hidden_size = (int)w_hh[0].size(0) / 3;

    int device_id = (int)x.device().index();
    cublasHandle_t handle = get_cublas_handle(device_id);

    // Bind cuBLAS to the current PyTorch CUDA stream.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id);
    cublasSetStream(handle, stream);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    // Working hidden state: (num_layers, batch, hidden)
    auto h = h0.clone();

    // Reusable scratch: preactivation buffers.
    auto pre_ih = torch::empty({batch_size, 3 * hidden_size}, opts);
    auto pre_hh = torch::empty({batch_size, 3 * hidden_size}, opts);

    // Inter-layer staging: full (seq, batch, hidden) buffer.
    // Upper layers read from this after layer l-1 writes each timestep.
    auto layer_input_buf = torch::empty({seq_len, batch_size, hidden_size}, opts);

    // Ensure weights are contiguous once.
    for (int l = 0; l < num_layers; l++) {
        w_ih[l] = w_ih[l].contiguous();
        w_hh[l] = w_hh[l].contiguous();
        b_ih[l] = b_ih[l].contiguous();
        b_hh[l] = b_hh[l].contiguous();
    }

    const int threads = 256;

    for (int l = 0; l < num_layers; l++) {
        int in_size = (l == 0) ? input_size : hidden_size;

        const float* w_ih_ptr = w_ih[l].data_ptr<float>();
        const float* w_hh_ptr = w_hh[l].data_ptr<float>();
        const float* b_ih_ptr = b_ih[l].data_ptr<float>();
        const float* b_hh_ptr = b_hh[l].data_ptr<float>();

        float* h_l = h.data_ptr<float>() + (long long)l * batch_size * hidden_size;

        for (int t = 0; t < seq_len; t++) {
            // x_t pointer: always (batch, in_size) row-major via normalized layout.
            const float* x_t;
            if (l == 0) {
                x_t = x_seq.data_ptr<float>() + (long long)t * batch_size * in_size;
            } else {
                x_t = layer_input_buf.data_ptr<float>() + (long long)t * batch_size * hidden_size;
            }

            // pre_ih = x_t @ W_ih^T
            sgemm_NT(handle, batch_size, 3*hidden_size, in_size,
                     x_t, w_ih_ptr, pre_ih.data_ptr<float>());
            // add b_ih
            {
                int n = batch_size * 3 * hidden_size;
                add_bias_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
                    pre_ih.data_ptr<float>(), b_ih_ptr, n, 3*hidden_size);
            }

            // pre_hh = h_l @ W_hh^T
            sgemm_NT(handle, batch_size, 3*hidden_size, hidden_size,
                     h_l, w_hh_ptr, pre_hh.data_ptr<float>());
            // add b_hh
            {
                int n = batch_size * 3 * hidden_size;
                add_bias_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
                    pre_hh.data_ptr<float>(), b_hh_ptr, n, 3*hidden_size);
            }

            // GRU cell update: h_l in-place
            {
                int n = batch_size * hidden_size;
                gru_cell_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
                    pre_ih.data_ptr<float>(),
                    pre_hh.data_ptr<float>(),
                    h_l,
                    batch_size, hidden_size);
            }

            // Stage output for next layer via same-stream copy kernel.
            if (l < num_layers - 1) {
                int n = batch_size * hidden_size;
                float* dst = layer_input_buf.data_ptr<float>() + (long long)t * batch_size * hidden_size;
                d2d_copy_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
                    dst, h_l, n);
            }
        }
    }

    return h;
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
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
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
