import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
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
    return f'stark_cuda_l3_p47_{digest}'

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

torch::Tensor vlad_norm_stage1(torch::Tensor vlad_bkd, torch::Tensor a_bdk, double eps);
torch::Tensor vlad_norm_stage2(torch::Tensor vlad_flat, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vlad_norm_stage1", &vlad_norm_stage1, "VLAD residual+normalize stage1 (CUDA)");
    m.def("vlad_norm_stage2", &vlad_norm_stage2, "VLAD row-normalize stage2 (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel 1: vlad is B x K x D, a is B x D x K
// Compute residual = vlad[b,k,d] - a[b,d,k], then L2-normalize over D for each (b,k)
// Output: B x D x K (contiguous, transposed layout)
__global__ void vlad_residual_normalize_kernel(
    const float* __restrict__ vlad,  // B x K x D
    const float* __restrict__ a,     // B x D x K
    float* __restrict__ out,         // B x D x K
    int B, int K, int D, float eps
) {
    // blockIdx.x = b, blockIdx.y = k
    int b = blockIdx.x;
    int k = blockIdx.y;
    if (b >= B || k >= K) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // vlad[b,k,d] offset: b*K*D + k*D + d
    const float* vlad_bk = vlad + b * K * D + k * D;
    // a[b,d,k] offset: b*D*K + d*K + k
    const float* a_bk = a + b * D * K + k;
    // out[b,d,k] offset: b*D*K + d*K + k
    float* out_bk = out + b * D * K + k;

    // Compute residuals and accumulate squared norm
    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += nthreads) {
        float v = vlad_bk[d] - a_bk[d * K];
        sdata[d] = v;  // store residual temporarily
        local_sum += v * v;
    }

    // Reduce squared norm across threads
    __shared__ float norm_buf[256];
    norm_buf[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) norm_buf[tid] += norm_buf[tid + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(fmaxf(norm_buf[0], eps));

    // Write normalized residuals in B x D x K layout
    for (int d = tid; d < D; d += nthreads) {
        out_bk[d * K] = sdata[d] * inv_norm;
    }
}

// Kernel 2: row-normalize flattened B x (D*K) tensor
__global__ void vlad_row_normalize_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int B, int DK, float eps
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* row = inp + b * DK;
    float* orow = out + b * DK;

    float local_sum = 0.0f;
    for (int i = tid; i < DK; i += nthreads) {
        float v = row[i];
        local_sum += v * v;
    }

    __shared__ float norm_buf[256];
    norm_buf[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) norm_buf[tid] += norm_buf[tid + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(fmaxf(norm_buf[0], eps));
    for (int i = tid; i < DK; i += nthreads) {
        orow[i] = row[i] * inv_norm;
    }
}

torch::Tensor vlad_norm_stage1(torch::Tensor vlad_bkd, torch::Tensor a_bdk, double eps) {
    TORCH_CHECK(vlad_bkd.is_cuda(), "vlad must be CUDA");
    TORCH_CHECK(a_bdk.is_cuda(), "a must be CUDA");
    TORCH_CHECK(vlad_bkd.is_contiguous(), "vlad must be contiguous");
    TORCH_CHECK(a_bdk.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(vlad_bkd.scalar_type() == torch::kFloat32, "vlad must be float32");
    TORCH_CHECK(a_bdk.scalar_type() == torch::kFloat32, "a must be float32");

    int B = vlad_bkd.size(0);
    int K = vlad_bkd.size(1);
    int D = vlad_bkd.size(2);

    // Output: B x D x K
    auto out = torch::empty({B, D, K}, vlad_bkd.options());

    // shared mem: D floats for residuals + 256 floats for norm reduction
    int threads = 256;
    size_t smem = (D + threads) * sizeof(float);

    dim3 grid(B, K);
    vlad_residual_normalize_kernel<<<grid, threads, smem>>>(
        vlad_bkd.data_ptr<float>(),
        a_bdk.data_ptr<float>(),
        out.data_ptr<float>(),
        B, K, D, (float)eps
    );

    return out;
}

torch::Tensor vlad_norm_stage2(torch::Tensor vlad_flat, double eps) {
    TORCH_CHECK(vlad_flat.is_cuda(), "vlad_flat must be CUDA");
    TORCH_CHECK(vlad_flat.is_contiguous(), "vlad_flat must be contiguous");
    TORCH_CHECK(vlad_flat.scalar_type() == torch::kFloat32, "vlad_flat must be float32");
    TORCH_CHECK(vlad_flat.dim() == 2, "vlad_flat must be 2D");

    int B = vlad_flat.size(0);
    int DK = vlad_flat.size(1);

    auto out = torch::empty_like(vlad_flat);

    int threads = 256;
    vlad_row_normalize_kernel<<<B, threads>>>(
        vlad_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        B, DK, (float)eps
    );

    return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters
        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size
        # <<<END_IMPROVE>>>

    def forward(self, x, mask=None):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """Aggregates feature maps into a fixed size representation.  In the following
                notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

                Args:
                    x (th.Tensor): B x N x D

                Returns:
                    (th.Tensor): B x DK
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        max_sample = x.size()[1]
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = x.view(-1, self.feature_size)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        if x.device != self.clusters.device:
                    msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
                    raise ValueError(msg)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        assignment = th.matmul(x, self.clusters)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        assignment = self.batch_norm(assignment)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        assignment = F.softmax(assignment, dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        assignment = assignment[:, :self.cluster_size]
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        assignment = assignment.view(-1, max_sample, self.cluster_size)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_10>>>
        a_sum = th.sum(assignment, dim=1, keepdim=True)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_11>>>
        a = a_sum * self.clusters2
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_12>>>
        # assignment remains B x N x K (no transpose needed)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_13>>>
        x = x.view(-1, max_sample, self.feature_size)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_14>>>
        vlad = th.matmul(x.transpose(1, 2), assignment)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_15>>>
        vlad = vlad if vlad.is_contiguous() else vlad.contiguous()
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_16>>>
        vlad = vlad - a
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_17>>>
        vlad = F.normalize(vlad)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_18>>>
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_19>>>
        vlad = F.normalize(vlad)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_20>>>
        return vlad
        # <<<END_IMPROVE>>>
