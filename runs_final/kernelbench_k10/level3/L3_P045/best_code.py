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
    return f'stark_cuda_l3_p45_{digest}'

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
    def __init__(self, in_channels, out_channels, features):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        torch.backends.cudnn.benchmark = True
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck =DoubleConv(features * 8, features * 16)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self._cg = None
        self._cg_static_in = None
        self._cg_static_out = None
        self._cg_shape = None
        self._cg_dtype = None
        self._cg_device = None
        # <<<END_IMPROVE>>>

    def forward(self, x):
        # <<<IMPROVE:forward_stmt_1>>>
        if x.is_cuda and not self.training:
            sig_shape = tuple(x.shape)
            sig_dtype = x.dtype
            sig_device = x.device
            if (self._cg is None or self._cg_shape != sig_shape or
                    self._cg_dtype != sig_dtype or self._cg_device != sig_device):
                self._cg_static_in = x.clone()
                with torch.no_grad():
                    s_in = self._cg_static_in
                    _e1 = self.encoder1(s_in)
                    _e2 = self.encoder2(self.pool1(_e1))
                    _e3 = self.encoder3(self.pool2(_e2))
                    _e4 = self.encoder4(self.pool3(_e3))
                    _bn = self.bottleneck(self.pool4(_e4))
                    _d4 = self.upconv4(_bn)
                    _d4 = torch.cat((_d4, _e4), dim=1)
                    _d4 = self.decoder4(_d4)
                    _d3 = self.upconv3(_d4)
                    _d3 = torch.cat((_d3, _e3), dim=1)
                    _d3 = self.decoder3(_d3)
                    _d2 = self.upconv2(_d3)
                    _d2 = torch.cat((_d2, _e2), dim=1)
                    _d2 = self.decoder2(_d2)
                    _d1 = self.upconv1(_d2)
                    _d1 = torch.cat((_d1, _e1), dim=1)
                    _d1 = self.decoder1(_d1)
                    warmup_out = self.final_conv(_d1)
                self._cg_static_out = torch.empty_like(warmup_out)
                g = torch.cuda.CUDAGraph()
                self._cg_static_in.copy_(x)
                with torch.cuda.graph(g):
                    _e1 = self.encoder1(self._cg_static_in)
                    _e2 = self.encoder2(self.pool1(_e1))
                    _e3 = self.encoder3(self.pool2(_e2))
                    _e4 = self.encoder4(self.pool3(_e3))
                    _bn = self.bottleneck(self.pool4(_e4))
                    _d4 = self.upconv4(_bn)
                    _d4 = torch.cat((_d4, _e4), dim=1)
                    _d4 = self.decoder4(_d4)
                    _d3 = self.upconv3(_d4)
                    _d3 = torch.cat((_d3, _e3), dim=1)
                    _d3 = self.decoder3(_d3)
                    _d2 = self.upconv2(_d3)
                    _d2 = torch.cat((_d2, _e2), dim=1)
                    _d2 = self.decoder2(_d2)
                    _d1 = self.upconv1(_d2)
                    _d1 = torch.cat((_d1, _e1), dim=1)
                    _d1 = self.decoder1(_d1)
                    _graph_out = self.final_conv(_d1)
                    self._cg_static_out.copy_(_graph_out)
                self._cg = g
                self._cg_shape = sig_shape
                self._cg_dtype = sig_dtype
                self._cg_device = sig_device
            self._cg_static_in.copy_(x, non_blocking=True)
            self._cg.replay()
            return self._cg_static_out
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        enc1 = self.encoder1(x)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_3>>>
        enc2 = self.encoder2(self.pool1(enc1))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_4>>>
        enc3 = self.encoder3(self.pool2(enc2))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_5>>>
        enc4 = self.encoder4(self.pool3(enc3))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_6>>>
        bottleneck = self.bottleneck(self.pool4(enc4))
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_7>>>
        dec4 = self.upconv4(bottleneck)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_8>>>
        dec4 = torch.cat((dec4, enc4), dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_9>>>
        dec4 = self.decoder4(dec4)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_10>>>
        dec3 = self.upconv3(dec4)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_11>>>
        dec3 = torch.cat((dec3, enc3), dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_12>>>
        dec3 = self.decoder3(dec3)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_13>>>
        dec2 = self.upconv2(dec3)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_14>>>
        dec2 = torch.cat((dec2, enc2), dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_15>>>
        dec2 = self.decoder2(dec2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_16>>>
        dec1 = self.upconv1(dec2)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_17>>>
        dec1 = torch.cat((dec1, enc1), dim=1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_18>>>
        dec1 = self.decoder1(dec1)
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_19>>>
        return self.final_conv(dec1)
        # <<<END_IMPROVE>>>
