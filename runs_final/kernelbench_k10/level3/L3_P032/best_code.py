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
    return f'stark_cuda_l3_p32_{digest}'

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

// Add pybind exports for custom CUDA entrypoints here.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
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
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6,:
        super().__init__()
        # <<<IMPROVE:init_body>>>
        """
                Convolutional Vision Transformer (CViT) implementation.
                :param num_classes: Number of output classes for classification.
                :param embed_dim: Dimensionality of the embedding space.
                :param num_heads: Number of attention heads.
                :param num_layers: Number of transformer layers.
                :param mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
                :param patch_size: Size of the convolutional patches.
                :param in_channels: Number of input channels (e.g., 3 for RGB images).
                :param image_size: Height/width of the square input image.
                """
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)
        self.transformer_layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=int(embed_dim * mlp_ratio),
                        dropout=0.0,
                        batch_first=True
                    ) for _ in range(num_layers)
                ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Forward pass of the CViT model.
                :param x: Input tensor of shape (B, C, H, W)
                :return: Output tensor of shape (B, num_classes)
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        B = x.size(0)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        x = self.conv1(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        x = x.flatten(start_dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        x = self.linear_proj(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        for layer in self.transformer_layers:
                    x = layer(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        return self.fc_out(x[:, 0])
        # <<<END_IMPROVE>>>
