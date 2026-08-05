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
    return f'stark_cuda_l3_p46_{digest}'

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

torch::Tensor netvlad_transpose_subtract_cuda(torch::Tensor vlad_bkd, torch::Tensor a_bdk);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("netvlad_transpose_subtract_cuda", &netvlad_transpose_subtract_cuda, "Fused NetVLAD transpose+subtract CUDA");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void netvlad_transpose_subtract_kernel(
    const float* __restrict__ vlad_bkd,
    const float* __restrict__ a_bdk,
    float* __restrict__ out,
    int B, int K, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * K;
    if (idx < total) {
        int k = idx % K;
        int d = (idx / K) % D;
        int b = idx / (K * D);
        int src_offset = (b * K + k) * D + d;
        out[idx] = vlad_bkd[src_offset] - a_bdk[idx];
    }
}

torch::Tensor netvlad_transpose_subtract_cuda(torch::Tensor vlad_bkd, torch::Tensor a_bdk) {
    TORCH_CHECK(vlad_bkd.is_cuda(), "vlad_bkd must be a CUDA tensor");
    TORCH_CHECK(a_bdk.is_cuda(), "a_bdk must be a CUDA tensor");
    TORCH_CHECK(vlad_bkd.is_contiguous(), "vlad_bkd must be contiguous");
    TORCH_CHECK(a_bdk.is_contiguous(), "a_bdk must be contiguous");
    TORCH_CHECK(vlad_bkd.dtype() == torch::kFloat32, "vlad_bkd must be float32");
    TORCH_CHECK(a_bdk.dtype() == torch::kFloat32, "a_bdk must be float32");
    TORCH_CHECK(vlad_bkd.dim() == 3, "vlad_bkd must be 3D [B,K,D]");
    TORCH_CHECK(a_bdk.dim() == 3, "a_bdk must be 3D [B,D,K]");

    int B = vlad_bkd.size(0);
    int K = vlad_bkd.size(1);
    int D = vlad_bkd.size(2);

    TORCH_CHECK(a_bdk.size(0) == B, "batch mismatch");
    TORCH_CHECK(a_bdk.size(1) == D, "feature dim mismatch");
    TORCH_CHECK(a_bdk.size(2) == K, "cluster dim mismatch");

    auto out = torch::empty_like(a_bdk);
    int numel = B * D * K;
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    netvlad_transpose_subtract_kernel<<<blocks, threads>>>(
        vlad_bkd.data_ptr<float>(),
        a_bdk.data_ptr<float>(),
        out.data_ptr<float>(),
        B, K, D
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
        assignment = assignment.transpose(1, 2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_13>>>
        x = x.view(-1, max_sample, self.feature_size)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_14>>>
        vlad = th.matmul(assignment, x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_15>>>
        pass
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_16>>>
        vlad = _stark_get_extension().netvlad_transpose_subtract_cuda(vlad.contiguous(), a.contiguous())
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
