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
    return f'stark_cuda_l3_p33_{digest}'

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

torch::Tensor fused_i2h_tanh(torch::Tensor x, torch::Tensor hidden,
                              torch::Tensor weight, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_i2h_tanh", &fused_i2h_tanh, "Fused cat+i2h+tanh (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Register-blocked fused kernel: each thread computes REG_N output columns
// Block: (TILE_N/REG_N, TILE_M) threads, each thread covers REG_N columns
#define TILE_M 16
#define TILE_N 64
#define TILE_K 32
#define REG_N  4   // each thread accumulates REG_N output columns
// block dims: (TILE_N/REG_N, TILE_M) = (16, 16) = 256 threads

__global__ void fused_i2h_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hidden,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M, int K_x, int K_h, int N)
{
    // threadIdx.x in [0, TILE_N/REG_N), threadIdx.y in [0, TILE_M)
    int thread_col_base = (blockIdx.x * TILE_N) + threadIdx.x * REG_N;
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int K_total = K_x + K_h;

    // Shared memory: sA holds a TILE_M x TILE_K slice of virtual-concat input
    //                sB holds a TILE_K x TILE_N slice of weight
    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    float acc[REG_N];
    #pragma unroll
    for (int r = 0; r < REG_N; r++) acc[r] = 0.0f;

    // Total threads per block = (TILE_N/REG_N) * TILE_M
    int tid = threadIdx.y * (TILE_N / REG_N) + threadIdx.x;
    int block_threads = TILE_M * (TILE_N / REG_N); // = 256

    int num_tiles = (K_total + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * TILE_K;

        // Load sA: TILE_M rows x TILE_K cols, flattened across all threads
        // TILE_M * TILE_K = 16*32 = 512 elements, block_threads=256, so 2 elements per thread
        for (int idx = tid; idx < TILE_M * TILE_K; idx += block_threads) {
            int r = idx / TILE_K;
            int k = idx % TILE_K;
            int global_row = blockIdx.y * TILE_M + r;
            int k_a = k_base + k;
            float val = 0.0f;
            if (global_row < M && k_a < K_total) {
                if (k_a < K_x) {
                    val = x[global_row * K_x + k_a];
                } else {
                    val = hidden[global_row * K_h + (k_a - K_x)];
                }
            }
            sA[r][k] = val;
        }

        // Load sB: TILE_K rows x TILE_N cols, flattened
        // TILE_K * TILE_N = 32*64 = 2048 elements, block_threads=256, so 8 elements per thread
        for (int idx = tid; idx < TILE_K * TILE_N; idx += block_threads) {
            int k = idx / TILE_N;
            int n = idx % TILE_N;
            int k_b = k_base + k;
            int n_b = blockIdx.x * TILE_N + n;
            float val = 0.0f;
            if (n_b < N && k_b < K_total) {
                // weight layout: [N, K_total] (nn.Linear stores [out_features, in_features])
                val = weight[n_b * K_total + k_b];
            }
            sB[k][n] = val;
        }

        __syncthreads();

        // Compute: thread (threadIdx.x, threadIdx.y) accumulates
        // row=blockIdx.y*TILE_M+threadIdx.y, cols thread_col_base..thread_col_base+REG_N-1
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a = sA[threadIdx.y][k];
            int local_col = threadIdx.x * REG_N;
            #pragma unroll
            for (int r = 0; r < REG_N; r++) {
                acc[r] += a * sB[k][local_col + r];
            }
        }

        __syncthreads();
    }

    // Write output with bias + tanh
    if (row < M) {
        #pragma unroll
        for (int r = 0; r < REG_N; r++) {
            int col = thread_col_base + r;
            if (col < N) {
                out[row * N + col] = tanhf(acc[r] + bias[col]);
            }
        }
    }
}

torch::Tensor fused_i2h_tanh(torch::Tensor x, torch::Tensor hidden,
                              torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(hidden.is_cuda(), "hidden must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32, "hidden must be float32");

    int M = x.size(0);
    int K_x = x.size(1);
    int K_h = hidden.size(1);
    int N = weight.size(0);

    auto out = torch::empty({M, N}, x.options());

    // block: (TILE_N/REG_N, TILE_M) = (16, 16)
    dim3 block(TILE_N / REG_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    fused_i2h_tanh_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        hidden.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K_x, K_h, N
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Initialize the Vanilla RNN model.

                :param input_size: The number of input features (int).
                :param hidden_size: The size of the hidden state (int).
                :param output_size: The number of output features (int).
                """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the Vanilla RNN.

                :param x: Input tensor of shape (batch_size, input_size).
                :param hidden: Hidden state tensor of shape (batch_size, hidden_size).
                :return: Output tensor of shape (batch_size, output_size), and the new hidden state.
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if initial_hidden is not None:
                    self.hidden.copy_(initial_hidden)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        self.hidden = self.hidden.to(x.device)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        combined = torch.cat((x, self.hidden), dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        self.hidden = self.tanh(self.i2h(combined))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        output = self.h2o(self.hidden)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        return output
        # <<<END_IMPROVE>>>
