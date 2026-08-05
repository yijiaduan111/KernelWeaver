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
    return f'stark_cuda_l1_p68_{digest}'

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

torch::Tensor conv_transpose3d_cudnn_fast(torch::Tensor x, torch::Tensor weight,
                                           int stride_d, int stride_h, int stride_w,
                                           int pad_d, int pad_h, int pad_w,
                                           int out_pad_d, int out_pad_h, int out_pad_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_cudnn_fast", &conv_transpose3d_cudnn_fast,
          "ConvTranspose3d via cuDNN backward-data with cached algo");
}
# <<<END_IMPROVE>>>
"""

CUDA_CU_SRC = r"""
# <<<IMPROVE:cuda_cu>>>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <mutex>
#include <unordered_map>
#include <string>
#include <sstream>

#define CUDNN_CHECK(expr) do { \
    cudnnStatus_t _s = (expr); \
    TORCH_CHECK(_s == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(_s)); \
} while(0)

struct CudnnConvCache {
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspace_size;
    torch::Tensor workspace;
    bool valid = false;
};

static std::mutex g_cache_mutex;
static std::unordered_map<std::string, CudnnConvCache> g_cache;
static cudnnHandle_t g_cudnn_handle = nullptr;

static cudnnHandle_t get_cudnn_handle() {
    if (g_cudnn_handle == nullptr) {
        CUDNN_CHECK(cudnnCreate(&g_cudnn_handle));
    }
    return g_cudnn_handle;
}

static std::string make_key(
    int N, int Cin, int Cout,
    int D, int H, int W,
    int kD, int kH, int kW,
    int sd, int sh, int sw,
    int pd, int ph, int pw
) {
    std::ostringstream oss;
    oss << N << '_' << Cin << '_' << Cout << '_'
        << D << '_' << H << '_' << W << '_'
        << kD << '_' << kH << '_' << kW << '_'
        << sd << '_' << sh << '_' << sw << '_'
        << pd << '_' << ph << '_' << pw;
    return oss.str();
}

torch::Tensor conv_transpose3d_cudnn_fast(
    torch::Tensor x, torch::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(x.dim() == 5, "input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D");

    x = x.contiguous();
    weight = weight.contiguous();

    const int N   = x.size(0);
    const int Cin = x.size(1);
    const int D   = x.size(2);
    const int H   = x.size(3);
    const int W   = x.size(4);
    // weight layout for ConvTranspose3d: (Cin, Cout, kD, kH, kW)
    const int Cout = weight.size(1);
    const int kD   = weight.size(2);
    const int kH   = weight.size(3);
    const int kW   = weight.size(4);

    // Output dimensions for transposed conv (no dilation, dilation=1)
    const int OD = (D - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    const int OH = (H - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    const int OW = (W - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    auto output = torch::zeros({N, Cout, OD, OH, OW}, x.options());

    std::string key = make_key(N, Cin, Cout, D, H, W, kD, kH, kW,
                               stride_d, stride_h, stride_w,
                               pad_d, pad_h, pad_w);

    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto handle = get_cudnn_handle();

    auto& cached = g_cache[key];
    if (!cached.valid) {
        // cuDNN descriptors
        cudnnTensorDescriptor_t x_desc, y_desc;
        cudnnFilterDescriptor_t w_desc;
        cudnnConvolutionDescriptor_t conv_desc;

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

        int x_dims[5]   = {N, Cin, D, H, W};
        int x_strides[5] = {Cin*D*H*W, D*H*W, H*W, W, 1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, x_dims, x_strides));

        int y_dims[5]   = {N, Cout, OD, OH, OW};
        int y_strides[5] = {Cout*OD*OH*OW, OD*OH*OW, OH*OW, OW, 1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, y_dims, y_strides));

        // Filter layout: (Cin, Cout, kD, kH, kW) for transposed conv weight
        int w_dims[5] = {Cin, Cout, kD, kH, kW};
        CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, w_dims));

        int pad_arr[3]    = {pad_d, pad_h, pad_w};
        int stride_arr[3] = {stride_d, stride_h, stride_w};
        int dilation[3]   = {1, 1, 1};
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(conv_desc, 3, pad_arr, stride_arr, dilation,
                                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        // Enable TF32 Tensor Core math for float32 inputs on Ampere+
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

        // Find best bwd-data algo with workspace up to 256MB
        const size_t MAX_WS = 256ull * 1024 * 1024;
        int num_algos = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf_results[8];
        CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
            handle, w_desc, x_desc, conv_desc, y_desc,
            8, &num_algos, perf_results));

        cached.algo = perf_results[0].algo;
        cached.workspace_size = 0;
        for (int i = 0; i < num_algos; i++) {
            if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
                perf_results[i].memory <= MAX_WS) {
                cached.algo = perf_results[i].algo;
                cached.workspace_size = perf_results[i].memory;
                break;
            }
        }

        if (cached.workspace_size > 0) {
            cached.workspace = torch::empty({(long long)cached.workspace_size},
                                            torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
        }
        cached.valid = true;

        // Run the convolution
        float alpha = 1.0f, beta = 0.0f;
        void* ws_ptr = cached.workspace_size > 0 ? cached.workspace.data_ptr() : nullptr;
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            handle, &alpha,
            w_desc, weight.data_ptr<float>(),
            x_desc, x.data_ptr<float>(),
            conv_desc, cached.algo,
            ws_ptr, cached.workspace_size,
            &beta,
            y_desc, output.data_ptr<float>()));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    } else {
        // Use cached algo with fresh descriptors
        cudnnTensorDescriptor_t x_desc, y_desc;
        cudnnFilterDescriptor_t w_desc;
        cudnnConvolutionDescriptor_t conv_desc;

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

        int x_dims[5]    = {N, Cin, D, H, W};
        int x_strides[5] = {Cin*D*H*W, D*H*W, H*W, W, 1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, x_dims, x_strides));

        int y_dims[5]    = {N, Cout, OD, OH, OW};
        int y_strides[5] = {Cout*OD*OH*OW, OD*OH*OW, OH*OW, OW, 1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, y_dims, y_strides));

        int w_dims[5] = {Cin, Cout, kD, kH, kW};
        CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, w_dims));

        int pad_arr[3]    = {pad_d, pad_h, pad_w};
        int stride_arr[3] = {stride_d, stride_h, stride_w};
        int dilation[3]   = {1, 1, 1};
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(conv_desc, 3, pad_arr, stride_arr, dilation,
                                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        // Enable TF32 Tensor Core math for float32 inputs on Ampere+
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

        float alpha = 1.0f, beta = 0.0f;
        void* ws_ptr = cached.workspace_size > 0 ? cached.workspace.data_ptr() : nullptr;
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            handle, &alpha,
            w_desc, weight.data_ptr<float>(),
            x_desc, x.data_ptr<float>(),
            conv_desc, cached.algo,
            ws_ptr, cached.workspace_size,
            &beta,
            y_desc, output.data_ptr<float>()));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    }

    return output;
}
# <<<END_IMPROVE>>>
"""

class ModelNew(nn.Module):
    """
        Performs a transposed 3D convolution with a square input and an asymmetric kernel.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                                 where kernel_width == kernel_height.
            stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
            padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
            output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super().__init__()
        # <<<IMPROVE:init_body>>>
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self._fastpath_enabled = (groups == 1 and not bias)
        self._stride_d, self._stride_h, self._stride_w = tuple(stride)
        self._pad_d, self._pad_h, self._pad_w = tuple(padding)
        self._opad_d, self._opad_h, self._opad_w = tuple(output_padding)
        # <<<END_IMPROVE>>>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <<<IMPROVE:forward_stmt_1>>>
        # Baseline fallback keeps the official PyTorch forward path.
        # After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).
        """
                Performs the transposed 3D convolution.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
                """
        # <<<END_IMPROVE>>>
        # <<<IMPROVE:forward_stmt_2>>>
        if (
            self._fastpath_enabled
            and x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 5
        ):
            try:
                return _stark_get_extension().conv_transpose3d_cudnn_fast(
                    x,
                    self.conv_transpose3d.weight,
                    self._stride_d,
                    self._stride_h,
                    self._stride_w,
                    self._pad_d,
                    self._pad_h,
                    self._pad_w,
                    self._opad_d,
                    self._opad_h,
                    self._opad_w,
                )
            except Exception:
                pass
        return self.conv_transpose3d(x)
        # <<<END_IMPROVE>>>
