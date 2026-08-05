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
    return f'stark_cuda_l3_p28_{digest}'

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

torch::Tensor prepend_cls_and_add_pos_cuda(
    torch::Tensor x,
    torch::Tensor cls_token,
    torch::Tensor pos_embedding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepend_cls_and_add_pos", &prepend_cls_and_add_pos_cuda,
        "Fuse cls prepend and positional embedding add (CUDA)");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void prepend_cls_and_add_pos_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cls_token,
    const scalar_t* __restrict__ pos_embedding,
    scalar_t* __restrict__ out,
    int B,
    int N,
    int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * (N + 1) * D;
  if (idx >= total) {
    return;
  }

  int d = idx % D;
  int tmp = idx / D;
  int n = tmp % (N + 1);
  int b = tmp / (N + 1);

  scalar_t value = n == 0 ? cls_token[d] : x[(b * N + (n - 1)) * D + d];
  out[idx] = value + pos_embedding[n * D + d];
}

}  // namespace

torch::Tensor prepend_cls_and_add_pos_cuda(
    torch::Tensor x,
    torch::Tensor cls_token,
    torch::Tensor pos_embedding) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(cls_token.is_cuda(), "cls_token must be a CUDA tensor");
  TORCH_CHECK(pos_embedding.is_cuda(), "pos_embedding must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B, N, D]");
  TORCH_CHECK(cls_token.dim() == 3, "cls_token must have shape [1, 1, D]");
  TORCH_CHECK(pos_embedding.dim() == 3, "pos_embedding must have shape [1, N+1, D]");
  TORCH_CHECK(cls_token.size(0) == 1 && cls_token.size(1) == 1, "cls_token must have shape [1, 1, D]");
  TORCH_CHECK(pos_embedding.size(0) == 1, "pos_embedding batch dimension must be 1");
  TORCH_CHECK(x.size(2) == cls_token.size(2), "embedding dims must match");
  TORCH_CHECK(pos_embedding.size(1) == x.size(1) + 1, "pos_embedding sequence dimension must equal N + 1");
  TORCH_CHECK(pos_embedding.size(2) == x.size(2), "pos_embedding embedding dim must match x");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto x_contig = x.contiguous();
  auto cls_contig = cls_token.contiguous();
  auto pos_contig = pos_embedding.contiguous();

  const int B = static_cast<int>(x_contig.size(0));
  const int N = static_cast<int>(x_contig.size(1));
  const int D = static_cast<int>(x_contig.size(2));

  auto out = torch::empty({B, N + 1, D}, x_contig.options());
  const int total = B * (N + 1) * D;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_contig.scalar_type(), "prepend_cls_and_add_pos_cuda", [&] {
    prepend_cls_and_add_pos_kernel<scalar_t><<<blocks, threads>>>(
        x_contig.data_ptr<scalar_t>(),
        cls_contig.data_ptr<scalar_t>(),
        pos_contig.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        B,
        N,
        D);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Vision Transformer (ViT) model.

                :param image_size: The size of the input image (assumed to be square).
                :param patch_size: The size of each patch (assumed to be square).
                :param num_classes: The number of output classes.
                :param dim: The dimensionality of the embedding space.
                :param depth: The number of transformer layers.
                :param heads: The number of attention heads.
                :param mlp_dim: The dimensionality of the MLP (Multi-Layer Perceptron) in the transformer.
                :param channels: The number of channels in the input image (default is 3 for RGB).
                :param dropout: Dropout rate applied in the MLP.
                :param emb_dropout: Dropout rate applied to the embedded patches.
                """
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
                    num_layers=depth
                )
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, num_classes)
                )
        # <<<END_IMPROVE>>>

    def forward(self, img):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the Vision Transformer.

                :param img: The input image tensor, shape (batch_size, channels, image_size, image_size).
                :return: The output tensor, shape (batch_size, num_classes).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        p = self.patch_size
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = self.patch_to_embedding(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = _stark_get_extension().prepend_cls_and_add_pos(x, self.cls_token, self.pos_embedding)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        x = x
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        x = self.dropout(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        x = self.transformer(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_10>>>
        x = self.to_cls_token(x[:, 0])
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_11>>>
        return self.mlp_head(x)
        # <<<END_IMPROVE>>>
