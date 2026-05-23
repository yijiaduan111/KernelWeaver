# Run Summary

- Run dir: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_claude_delib_main_20260513_124238`
- Tasks with run.json: `1`
- Summary rows: `15`
- task_count: `15`
- success_count: `3`
- compile_rate: `1.0`
- correct_rate: `0.13333333333333333`
- best_speedup: `1.0304878048780488`
- median_speedup: `1.0`
- improved_over_reference_rate: `0.06666666666666667`
- paper_metrics: `{"Fast1": 0.13333333333333333, "Speed": 0.20119570024152125, "Success": 0.2}`

| Task | Speedup | Best | Attempts | Compile OK | Correct | Strategies | Op Type |
|---|---:|---|---:|---:|---:|---:|---|
| `L1_P25_Swish_l1_p25` | `None` | `root` | `10` | `11` | `0` | `9` | `elementwise` |

# L1_P25_Swish_l1_p25

- Task name: `kernelbench_l1_25_25_swish`
- Level/problem: `1/25`
- Backend: `cuda`
- Best node: `root`
- Speedup: `None`
- Source origin: `KernelBench/KernelBench/level1/25_Swish.py`

## Semantic Profile

- Enabled: `True`
- Mode: `rule`
- Op type: `elementwise`
- Summary: Elementwise computation; likely optimization is fusing pointwise math and avoiding intermediate tensors.
- Recommended anchors: `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]`
- Risk notes: `["preserve exact activation math and broadcasting", "check fast-math numerical tolerance"]`

| # | Intent | Priority | Target Anchors | Summary |
|---:|---|---:|---|---|
| 1 | `fuse_elementwise_ops` | `5` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]` | fuse elementwise ops |
| 2 | `avoid_intermediate_allocations` | `4` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]` | avoid intermediate allocations |

Anchor hints:
- `forward_stmt_1`: {"anchor_name": "forward_stmt_1", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site", "use one thread per element", "prefer a grid-stride loop"], "op_names": ["_swish", "a", "activation", "any", "applied", "applies", "args", "as", "input", "kernelbench", "level1", "model"], "optimization_intents": ["fuse_elementwise_ops", "avoid_intermediate_allocations"], "priority": 5, "region_role": "forward", "risk_notes": ["preserve exact activation math a...
- `forward_stmt_2`: {"anchor_name": "forward_stmt_2", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site", "use one thread per element", "prefer a grid-stride loop"], "op_names": ["_swish", "a", "activation", "any", "applied", "applies", "args", "arithmetic", "as", "input", "kernelbench", "level1"], "optimization_intents": ["fuse_elementwise_ops", "avoid_intermediate_allocations"], "priority": 5, "region_role": "forward", "risk_notes": ["preserve exact activation m...
- `cuda_cpp`: {"anchor_name": "cuda_cpp", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"], "op_names": ["add", "cuda", "custom", "entrypoints", "exports", "extension", "for", "h", "here", "include", "m", "pybind"], "optimization_intents": ["implement_backend_kernel"], "priority": 4, "region_role": "helper", "risk_notes": ["inspect shape, dtype, and broadcasting assumptions before editing"], "semantic_type": "backend_kernel_region", "source_excerpt": "#in...
- `cuda_cu`: {"anchor_name": "cuda_cu", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"], "op_names": ["add", "and", "cuda", "cuda_runtime", "exported", "extension", "functions", "h", "here", "include", "kernels", "torch"], "optimization_intents": ["implement_backend_kernel"], "priority": 4, "region_role": "helper", "risk_notes": ["inspect shape, dtype, and broadcasting assumptions before editing"], "semantic_type": "backend_kernel_region", "source_excer...
- `helpers`: {"anchor_name": "helpers", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"], "op_names": ["_stark_extension", "_stark_extension_name", "_stark_get_extension", "_stark_strip_anchor_markers", "append", "arithmetic", "cleaned_lines", "cleaned_lines.append", "continue", "cpp_sources", "cuda_cpp_src", "cuda_cu_src"], "optimization_intents": ["inspect_source_before_editing"], "priority": 2, "region_role": "helper", "risk_notes": ["inspect shape, d...
- `init_body`: {"anchor_name": "init_body", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"], "op_names": [], "optimization_intents": ["inspect_source_before_editing"], "priority": 2, "region_role": "init", "risk_notes": ["inspect shape, dtype, and broadcasting assumptions before editing"], "semantic_type": "state_initialization", "source_excerpt": ""}

## Strategy Portfolio

- Enabled: `True`
- Mode: `multi_model_v0`
- Providers: `["openai-compatible", "claude-compatible", "gemini-compatible"]`
- Proposal errors: `{}`
- Review errors: `{}`
- Strategy count: `9`

| ID | Intent | Source | Scores | Anchors | Summary |
|---|---|---|---|---|---|
| `strategy_01` | `fuse_elementwise_ops` | `["openai-compatible"]` | `{"claude-compatible": 5.0, "gemini-compatible": 5.0, "openai-compatible": 5.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a custom CUDA Swish kernel that computes y = x * sigmoid(x) in one pass, replacing the two-op PyTorch path and eliminating the intermediate sigmoid tensor. |
| `strategy_02` | `fuse_elementwise_swish_grid_stride` | `["claude-compatible"]` | `{"claude-compatible": 5.0, "gemini-compatible": 5.0, "openai-compatible": 5.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a fused Swish kernel (x * sigmoid(x)) in CUDA using a grid-stride loop, eliminating intermediate tensor allocations and leveraging fast-math intrinsics. |
| `strategy_03` | `fuse_elementwise_ops` | `["gemini-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a vectorized CUDA kernel using float4 for optimal memory bandwidth utilization. |
| `strategy_04` | `avoid_intermediate_allocations` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 4.0, "openai-compatible": 4.0}` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp"]` | Route contiguous CUDA inputs directly to a specialized backend kernel to avoid creating temporary tensors from torch.sigmoid(x) and to reduce allocator overhead in repeated benchmarking. |
| `strategy_05` | `vectorized_float4_swish_kernel` | `["claude-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Extend the fused Swish kernel to process 4 elements per thread using float4 vectorized loads/stores, improving memory bandwidth utilization. |
| `strategy_06` | `fuse_elementwise_ops` | `["gemini-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 4.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a basic CUDA kernel to fuse the multiplication and sigmoid operations into a single pass. |
| `strategy_08` | `fuse_elementwise_ops` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 3.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "helpers"]` | Specialize the CUDA implementation for the common floating-point path used in benchmarking, with simple launch configuration and contiguous-memory access to maximize throughput for this unary elementwise op. |
| `strategy_07` | `half_precision_swish_with_fp16_intrinsics` | `["claude-compatible"]` | `{"claude-compatible": 2.0, "gemini-compatible": 3.0, "openai-compatible": 2.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Add a float16 (half2) code path in the Swish kernel using CUDA half2 intrinsics to process two fp16 elements per instruction, doubling throughput on Tensor Core-capable GPUs. |
| `strategy_09` | `avoid_intermediate_allocations_inplace_output` | `["claude-compatible"]` | `{"claude-compatible": 2.0, "gemini-compatible": 2.0, "openai-compatible": 1.0}` | `["cuda_cu", "cuda_cpp", "init_body", "forward_stmt_1", "forward_stmt_2"]` | Pre-allocate the output tensor once in the wrapper and reuse it across calls by using a persistent buffer strategy, avoiding repeated cudaMalloc overhead. |

Implementation hints and risks:
- `strategy_01` hints: ["Add a CUDA entrypoint exported through the extension and invoke it from the forward path while keeping ModelNew I/O unchanged.", "Use one thread per element over a flattened tensor and a grid-stride loop for arbitrary shapes.", "Allocate exactly one output tensor and compute sigmoid plus multiply inline inside the kernel.", "Prefer the custom path for CUDA tensors and preserve a PyTorch fallback for unsupported cases."]
- `strategy_01` risks: ["Preserve exact activation math and same-shape behavior for arbitrary input layouts.", "Check numerical tolerance because the scaffold already enables --use_fast_math."]
- `strategy_02` hints: ["Write a __global__ swish_kernel that takes float* input, float* output, and int64_t n.", "Use a grid-stride loop: for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x).", "Compute swish inline: float val = input[i]; output[i] = val / (1.0f + expf(-val));", "With --use_fast_math already set, expf will use fast approximation automatically.", "Expose a C++ wrapper swish_cuda(torch::Tensor x) that allocates output tensor, computes n = x.numel(), launches kernel with blockDim=256 and gridDim=min((n+255)/256, 65535).", "Register swish_cuda in PYBIND11_MODULE.", "In forward_stmt_1/forward_stmt_2, call _stark_get_extension().swish_cuda(x)."]
- `strategy_02` risks: ["Ensure input tensor is contiguous before passing pointer; call x.contiguous() in wrapper.", "Handle both float32 and float16 dtypes if needed; start with float32.", "fast-math expf may have slight numerical differences vs torch.sigmoid; verify correctness tolerance."]
- `strategy_03` hints: ["Cast the input and output data pointers to `float4*` to load and store 4 elements per thread.", "Compute the Swish activation for all 4 elements in registers.", "Include a scalar fallback loop at the end of the kernel to handle any remaining elements if the total size is not a multiple of 4.", "Use intrinsic math functions like `__expf` for maximum performance."]
- `strategy_03` risks: ["Requires careful handling of tensor sizes that are not multiples of the vector width.", "Pointer alignment must be guaranteed for safe vectorized memory accesses."]
- `strategy_04` hints: ["At the forward call site, dispatch to the extension only when x is on CUDA; otherwise keep the baseline expression.", "In the backend wrapper, validate dtype/device and construct an output tensor once before launching the kernel.", "Handle flattened numel-based processing so shape preservation is automatic without extra indexing structures.", "Keep launch wiring minimal in cuda_cpp/cuda_cu and avoid introducing extra staging buffers."]
- `strategy_04` risks: ["Inspect shape, dtype, and contiguity assumptions before editing the wrapper path.", "If non-contiguous tensors are possible, ensure correctness via fallback or explicit handling."]
- `strategy_05` hints: ["Cast input/output pointers to float4* for the bulk of the tensor (n/4 elements).", "Each thread loads one float4, applies swish to each component, stores one float4.", "Handle the tail (n % 4 elements) with a scalar epilogue loop.", "Keep blockDim=256; gridDim covers n/4 elements with grid-stride.", "Ensure tensor is 16-byte aligned (PyTorch allocations typically are).", "Use __ldg() for read-only input loads to go through texture cache."]
- `strategy_05` risks: ["Alignment must be guaranteed; add assertion or fallback for non-aligned cases.", "Tail handling must be correct to avoid out-of-bounds access.", "Only beneficial for large tensors; small tensors may not see improvement."]
- `strategy_06` hints: ["Flatten the input tensor to 1D using `.contiguous().view(-1)`.", "Write a CUDA kernel with a grid-stride loop to compute `x / (1.0f + expf(-x))` for each element.", "Allocate an empty output tensor of the same shape and populate it in the kernel.", "Expose the C++ wrapper via PyBind11 in `cuda_cpp` and replace the PyTorch operations in `forward_stmt_2`."]
- `strategy_06` risks: ["Ensure the input tensor is contiguous before passing its data pointer to CUDA.", "Slight numerical differences may occur due to `--use_fast_math`."]
- `strategy_08` hints: ["Focus the custom kernel on the benchmark-relevant floating dtypes first rather than a broad generic implementation.", "Use contiguous linear indexing and coalesced loads/stores; keep per-element work limited to the sigmoid and multiply.", "Leverage the existing extension build path in helpers and keep the exported interface narrow and task-specific.", "Tune block size conservatively for occupancy and use a standard grid-stride kernel rather than complex tiling."]
- `strategy_08` risks: ["Narrow dtype specialization can require fallback paths for unsupported types.", "Fast-math may slightly change sigmoid accuracy versus eager PyTorch."]
- `strategy_07` hints: ["Add a separate swish_kernel_fp16 that takes __half2* pointers.", "Use h2exp, __hmul2, __hadd2 intrinsics for half2 swish computation.", "In the C++ wrapper, dispatch on x.scalar_type(): kFloat -> float kernel, kHalf -> half2 kernel.", "Include <cuda_fp16.h> in cuda_cu.", "For half2 swish: val2 = input[i]; sigmoid2 = __h2div(__float2half2_rn(1.0f), __hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(val2)))); output[i] = __hmul2(val2, sigmoid2).", "Handle odd-length fp16 tensors with scalar tail."]
- `strategy_07` risks: ["half2 intrinsics require sm_53 or higher; add architecture guard.", "Numerical precision of half2 exp approximation must be validated.", "Only beneficial if benchmark actually uses fp16 tensors."]
- `strategy_09` hints: ["In the C++ wrapper, use torch::empty_like(x) for output allocation; this is already optimal for single-call use.", "Alternatively, expose a swish_cuda_out(torch::Tensor x, torch::Tensor out) variant that writes into a pre-allocated buffer.", "In init_body, pre-allocate a buffer of expected size using torch.empty and pass it to the out-variant in forward.", "This avoids repeated allocator calls when tensor shape is fixed across forward passes.", "Only implement if profiling shows allocation overhead is significant."]
- `strategy_09` risks: ["Buffer must be resized if input shape changes between calls; add shape check.", "Thread safety concerns if model is used in multi-threaded context.", "Benefit is marginal compared to the fused kernel strategy; implement only as a secondary optimization."]

## Strategy Usage By Attempt

| Node | Strategy | Correct | Compile | Speedup | Failure | Stage | Summary |
|---|---|---:|---:|---:|---|---|---|
| `n1` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n2` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n3` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n4` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n5` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n6` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n7` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n8` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n9` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |
| `n10` | `strategy_04` | `False` | `True` | `None` | `acceleratorerror` | `runtime` | Route CUDA inputs to a simple fused Swish backend kernel while preserving the eager PyTorch fallback, focusing on correctness and avoiding intermediate allocations after an illegal memory access failure. |

## Failure Breakdown

- Stats: `{"attempt_count": 10, "debug_attempts": 9, "failure_counts": {"acceleratorerror": 11}, "failure_stage_counts": {"runtime": 11}, "invalid_proposals": 0, "plan_attempts": 1, "pruned_count": 0}`
- Failure counts from nodes: `{'acceleratorerror': 11}`
- Stage counts from nodes: `{'runtime': 11}`

Recent node log snippets:
- `root` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n1` `acceleratorerror`: anchor_marker_drift: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']
- `n1` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n2` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n3` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n4` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n5` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n6` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n7` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n8` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n9` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- `n10` `acceleratorerror`: paper_eval_error: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

## Code Artifacts

- run.json: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_claude_delib_main_20260513_124238/L1_P25_Swish_l1_p25/run.json`
- best_code.py: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_claude_delib_main_20260513_124238/L1_P25_Swish_l1_p25/best_code.py`
- best_node_id: `root`
- best plan: -
- best strategy: `None`
