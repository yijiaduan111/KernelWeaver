# Run Summary

- Run dir: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447`
- Tasks with run.json: `1`
- Summary rows: `15`
- task_count: `15`
- success_count: `6`
- compile_rate: `0.5333333333333333`
- correct_rate: `0.36666666666666664`
- best_speedup: `2.1047120418848166`
- median_speedup: `1.0026018423401426`
- improved_over_reference_rate: `0.26666666666666666`
- paper_metrics: `{"Fast1": 0.4, "Speed": 0.5123897972437862, "Success": 0.4}`

| Task | Speedup | Best | Attempts | Compile OK | Correct | Strategies | Op Type |
|---|---:|---|---:|---:|---:|---:|---|
| `L1_P42_MaxPool2d_l1_p42` | `1.5759312320916905` | `n2` | `10` | `8` | `4` | `10` | `reduction` |

# L1_P42_MaxPool2d_l1_p42

- Task name: `kernelbench_l1_42_42_max_pooling_2d`
- Level/problem: `1/42`
- Backend: `cuda`
- Best node: `n2`
- Speedup: `1.5759312320916905`
- Source origin: `KernelBench/KernelBench/level1/42_Max_Pooling_2D.py`

## Semantic Profile

- Enabled: `True`
- Mode: `rule`
- Op type: `reduction`
- Summary: Reduction computation; likely optimization is preserving the reduced axis while using block/warp reductions.
- Recommended anchors: `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]`
- Risk notes: `["preserve reduction dimension and keepdim behavior", "handle non-divisible sizes and boundary masks"]`

| # | Intent | Priority | Target Anchors | Summary |
|---:|---|---:|---|---|
| 1 | `preserve_reduction_semantics` | `5` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]` | preserve reduction semantics |
| 2 | `use_block_or_warp_reduction` | `4` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]` | use block or warp reduction |

Anchor hints:
- `forward_stmt_1`: {"anchor_name": "forward_stmt_1", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site", "use block or warp reductions", "preserve the reduced dimension"], "op_names": ["_max_pooling_2d", "after", "applied", "applies", "args", "batch_size", "be", "before", "between", "channels", "d", "dilation"], "optimization_intents": ["preserve_reduction_semantics", "use_block_or_warp_reduction"], "priority": 5, "region_role": "forward", "risk_notes": ["preserv...
- `forward_stmt_2`: {"anchor_name": "forward_stmt_2", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site", "use block or warp reductions", "preserve the reduced dimension"], "op_names": ["_max_pooling_2d", "after", "applied", "applies", "args", "batch_size", "be", "before", "between", "channels", "d", "dilation"], "optimization_intents": ["preserve_reduction_semantics", "use_block_or_warp_reduction"], "priority": 5, "region_role": "forward", "risk_notes": ["preserv...
- `cuda_cpp`: {"anchor_name": "cuda_cpp", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"], "op_names": ["add", "cuda", "custom", "entrypoints", "exports", "extension", "for", "h", "here", "include", "m", "pybind"], "optimization_intents": ["implement_backend_kernel"], "priority": 4, "region_role": "helper", "risk_notes": ["inspect shape, dtype, and broadcasting assumptions before editing"], "semantic_type": "backend_kernel_region", "source_excerpt": "#in...
- `cuda_cu`: {"anchor_name": "cuda_cu", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"], "op_names": ["add", "and", "cuda", "cuda_runtime", "exported", "extension", "functions", "h", "here", "include", "kernels", "torch"], "optimization_intents": ["implement_backend_kernel"], "priority": 4, "region_role": "helper", "risk_notes": ["inspect shape, dtype, and broadcasting assumptions before editing"], "semantic_type": "backend_kernel_region", "source_excer...
- `helpers`: {"anchor_name": "helpers", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"], "op_names": ["_stark_extension", "_stark_extension_name", "_stark_get_extension", "_stark_strip_anchor_markers", "append", "arithmetic", "cleaned_lines", "cleaned_lines.append", "continue", "cpp_sources", "cuda_cpp_src", "cuda_cu_src"], "optimization_intents": ["inspect_source_before_editing"], "priority": 2, "region_role": "helper", "risk_notes": ["inspect shape, d...
- `init_body`: {"anchor_name": "init_body", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site"], "op_names": ["applied", "args", "be", "before", "between", "d", "dilation", "elements", "initializes", "int", "kernel", "kernel_size"], "optimization_intents": ["inspect_source_before_editing"], "priority": 2, "region_role": "init", "risk_notes": ["inspect shape, dtype, and broadcasting assumptions before editing"], "semantic_type": "state_initialization", "source...

## Strategy Portfolio

- Enabled: `True`
- Mode: `multi_model_v0`
- Providers: `["openai-compatible", "claude-compatible", "gemini-compatible"]`
- Proposal errors: `{}`
- Review errors: `{}`
- Strategy count: `10`

| ID | Intent | Source | Scores | Anchors | Summary |
|---|---|---|---|---|---|
| `strategy_07` | `optimize_branching` | `["gemini-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 5.0, "openai-compatible": 4.0}` | `["cuda_cu"]` | Precompute valid loop bounds to avoid boundary checks inside the pooling window loop. |
| `strategy_01` | `implement_direct_cuda_maxpool_forward` | `["openai-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 2.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Add a custom CUDA forward path for NCHW max-pooling that computes one output element per thread and reduces over the pooling window in registers, bypassing generic dispatcher overhead while preserving exact MaxPool2d output shape semantics. |
| `strategy_02` | `implement_backend_kernel` | `["gemini-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 2.0}` | `["cuda_cu", "cuda_cpp", "init_body", "forward_stmt_2"]` | Implement a custom CUDA kernel where each thread computes a single output element. |
| `strategy_03` | `specialize_for_common_pool_configs` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 3.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "init_body", "forward_stmt_1"]` | Introduce fast paths specialized for the most common Level-1 pooling parameter patterns, such as square kernels with stride equal to kernel size and no dilation, reducing index arithmetic and branch cost inside the CUDA kernel. |
| `strategy_04` | `Custom fused max-pool kernel with one thread per output element` | `["claude-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 1.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a CUDA kernel where each thread computes one output element by iterating over the kernel_size x kernel_size window. Use __ldg for read-only cache hints on input loads. Launch with a 2D grid covering (N*C, out_H*out_W) to maximize occupancy and avoid atomic operations entirely. |
| `strategy_06` | `optimize_occupancy` | `["gemini-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 4.0, "openai-compatible": 2.0}` | `["cuda_cu", "forward_stmt_2"]` | Flatten the 4D grid into a 1D grid to maximize block occupancy and simplify index calculations. |
| `strategy_05` | `Shared memory tiling to reuse input data across adjacent output elements` | `["claude-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 3.0, "openai-compatible": 2.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Load a tile of input data into shared memory covering the output tile plus halo region. Each thread block computes a TILE_H x TILE_W output patch, reusing shared memory loads across the kernel window iterations. Reduces global memory bandwidth for stride=1 cases. |
| `strategy_10` | `vectorize_memory_reads_when_layout_allows` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 2.0, "openai-compatible": 2.0}` | `["cuda_cu", "forward_stmt_1", "helpers"]` | Improve input read efficiency by using layout-aware vectorized loads or contiguous-channel/output traversal where pooling windows touch aligned memory, while retaining scalar fallback for edges and irregular dilation/padding cases. |
| `strategy_08` | `Warp-level reduction across kernel window using shuffle instructions` | `["claude-compatible"]` | `{"claude-compatible": 2.0, "gemini-compatible": 2.0, "openai-compatible": 1.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | For larger kernel sizes, assign a warp (32 threads) to each output element. Each thread handles a subset of the kernel window elements, then use __shfl_down_sync to reduce the max across the warp. This amortizes memory latency for large pooling windows. |
| `strategy_09` | `Vectorized float4 input loads with per-channel parallelism` | `["claude-compatible"]` | `{"claude-compatible": 2.0, "gemini-compatible": 2.0, "openai-compatible": 1.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Restructure the kernel to load input data using float4 vector loads when the width dimension is 4-aligned. Process multiple channels in the same thread block using the channel dimension as an inner loop, improving arithmetic intensity and hiding memory latency through instruction-level parallelism. |

Implementation hints and risks:
- `strategy_07` hints: ["Instead of checking `if (h_in >= 0 && h_in < H)` inside the innermost loops, calculate the valid start and end indices for the pooling window before the loop.", "Clamp the start and end indices to the valid input dimensions (0 to H or W).", "Iterate only over the valid range, eliminating branch divergence inside the loop.", "Initialize the max value to `-INFINITY` (or the lowest possible float) to handle cases where the window is entirely out of bounds (though padding usually prevents this)."]
- `strategy_07` risks: ["Careful calculation of loop bounds is required, especially when `dilation > 1`."]
- `strategy_01` hints: ["Expose a single CUDA entrypoint from cuda_cpp and bind it via pybind, keeping ModelNew I/O unchanged.", "In cuda_cu, map threads over flattened N*C*H_out*W_out output coordinates; each thread scans the kernel window and tracks the local maximum in registers.", "Precompute output dimensions and effective kernel extents from kernel_size/stride/padding/dilation exactly as nn.MaxPool2d does.", "In forward_stmt_1/2, dispatch to the extension only for CUDA contiguous tensors and supported dtypes; keep PyTorch fallback otherwise."]
- `strategy_01` risks: ["Preserve reduction semantics exactly, including padded regions and dilation-strided window traversal.", "Need careful boundary handling for non-divisible shapes and padded windows.", "Likely limited benefit if PyTorch already routes to heavily optimized cuDNN kernels on the benchmark shapes."]
- `strategy_02` hints: ["In `cuda_cu`, write a kernel taking input, output, and pooling parameters (kernel_size, stride, padding, dilation).", "Map each thread to an output element (N, C, H_out, W_out).", "Loop over the `kernel_size` window, computing input indices and maintaining the maximum value found.", "Export the function via pybind11 in `cuda_cpp`.", "In `init_body`, save the pooling parameters to `self`.", "In `forward_stmt_2`, compute the output shape using the standard MaxPool2D formula, allocate the output tensor, and invoke the custom CUDA extension."]
- `strategy_02` risks: ["Must correctly handle negative indices due to padding.", "Output shape calculation must exactly match PyTorch's formula to avoid shape mismatches."]
- `strategy_03` hints: ["Capture kernel_size/stride/padding/dilation scalars in init_body so forward can cheaply select a specialized entrypoint.", "Provide a generic kernel plus one or two specialized wrappers for common cases like 2x2/stride2/pad0 or 3x3/stride2/pad1.", "Simplify per-thread indexing in specialized kernels by hoisting invariant arithmetic and assuming fixed small window sizes.", "Keep the generic path available for unsupported parameter combinations to preserve correctness."]
- `strategy_03` risks: ["Benefit depends strongly on whether benchmark instances hit the specialized parameter sets.", "Too many variants can increase compile time and maintenance burden.", "Must not change output shape or semantics when switching between specialized and generic paths."]
- `strategy_04` hints: ["Map threadIdx.x to output spatial position (oh*out_W + ow) and blockIdx.y to (n*C + c)", "Use __ldg(&input[...]) for all input reads to exploit L1 read-only cache", "Handle padding by clamping or skipping out-of-bounds indices with -FLT_MAX sentinel", "Support dilation by computing ih = oh*stride - padding + kh*dilation", "Export via pybind11 as max_pool2d_forward; call from forward_stmt_2 replacing self.maxpool(x)", "Keep kernel_size, stride, padding, dilation as kernel arguments passed from init_body stored fields"]
- `strategy_04` risks: ["Must handle non-divisible spatial sizes correctly with boundary checks", "Dilation > 1 increases window span; ensure ih/iw bounds checks cover full dilated range", "Float vs half precision: start with float32 only"]
- `strategy_06` hints: ["Launch a 1D grid of threads where `total_threads = N * C * H_out * W_out`.", "Inside the kernel, compute `n, c, h_out, w_out` from `idx = blockIdx.x * blockDim.x + threadIdx.x` using integer division and modulo.", "This avoids 3D grid limits and ensures all blocks are fully populated, regardless of tensor dimensions."]
- `strategy_06` risks: ["Integer division and modulo can be computationally expensive; ensure the overhead is offset by the occupancy gains."]
- `strategy_05` hints: ["Choose TILE_H=8, TILE_W=16 output elements per block; shared memory tile is (TILE_H*stride + (kernel_size-1)*dilation) x (TILE_W*stride + (kernel_size-1)*dilation)", "Load input tile cooperatively using all threads in block before __syncthreads()", "Each thread then iterates over kernel window in shared memory to compute max", "Particularly effective when stride=1 and kernel_size=3 (9x reuse of each input element)", "Allocate shared memory dynamically based on actual tile dimensions passed at launch"]
- `strategy_05` risks: ["Shared memory size grows with dilation; may exceed 48KB for large kernels with dilation", "Stride > 1 reduces reuse factor; less beneficial for stride >= kernel_size", "Boundary handling at tensor edges requires careful padding logic in shared memory load"]
- `strategy_10` hints: ["Restrict the custom path to contiguous NCHW tensors and inspect tensor alignment/stride assumptions before using vectorized accesses.", "Arrange launch order so neighboring threads iterate adjacent output x positions, improving coalescing for overlapping windows.", "Use a scalar fallback for boundary outputs, odd widths, or parameter combinations that break alignment assumptions.", "Keep helper-side extension loading unchanged except for any additional exported symbol names."]
- `strategy_10` risks: ["Max-pooling windows often create irregular access under padding/dilation, limiting vectorization opportunities.", "Incorrect alignment assumptions can cause faults or silent performance regressions.", "Need strict guards so unsupported layouts fall back safely."]
- `strategy_08` hints: ["Assign one warp per output element: blockDim.x=32, each thread covers kernel_size*kernel_size/32 elements", "Use __shfl_down_sync(0xffffffff, val, offset) to reduce max across warp lanes", "Grid: (out_H*out_W + warps_per_block - 1)/warps_per_block blocks per (N,C) pair", "For kernel_size <= 5 (25 elements), pad to 32 with -FLT_MAX so warp reduction is clean", "Store result from lane 0 of each warp to output tensor"]
- `strategy_08` risks: ["Warp divergence if kernel_size*kernel_size is not a power of 2; pad with -FLT_MAX", "For small kernels (2x2=4 elements) this wastes 28 lanes; use strategy 1 instead", "Requires careful index mapping to avoid out-of-bounds reads"]
- `strategy_09` hints: ["Check if input width is divisible by 4; use float4 loads via reinterpret_cast<float4*>", "Assign blockIdx.z to batch*channels, blockIdx.x/y to output spatial tiles", "Use __builtin_expect for branch prediction on boundary checks", "Add --use_fast_math flag (already present in helpers) to enable fast fmaxf", "Use fmaxf() instead of max() for float comparisons to leverage hardware instruction", "Unroll the kernel window loop with #pragma unroll for small fixed kernel sizes"]
- `strategy_09` risks: ["float4 loads require 16-byte alignment; add alignment check or fallback to scalar loads", "Loop unrolling only works for compile-time constant kernel sizes; need template specialization", "Vectorization along width may conflict with padding boundary handling"]

## Strategy Usage By Attempt

| Node | Strategy | Correct | Compile | Speedup | Failure | Stage | Summary |
|---|---|---:|---:|---:|---|---|---|
| `n1` | `strategy_01` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Add a direct CUDA max-pool forward path for contiguous CUDA NCHW tensors, with one thread computing one output element and reducing the pooling window in registers, while keeping the existing PyTorch MaxPool2d fallback for unsupported cases. |
| `n2` | `strategy_01` | `True` | `True` | `1.5759312320916905` | `anchor_marker_drift` | `None` | Add a direct CUDA max-pool forward path for contiguous CUDA NCHW tensors, with one thread computing one output element and reducing the pooling window in registers, while keeping the existing PyTorch MaxPool2d fallback for unsupported cases. |
| `n3` | `strategy_07` | `False` | `True` | `None` | `builtins.RuntimeError` | `runtime` | Reduce branch divergence in the custom CUDA max-pooling kernel by precomputing valid kernel-loop bounds for each output element, so the inner reduction loop only iterates over in-bounds window positions. |
| `n4` | `strategy_06` | `False` | `True` | `None` | `builtins.RuntimeError` | `runtime` | Keep the custom CUDA max-pooling path but simplify and harden the launch/indexing by editing only the CUDA kernel region to use a clear flat 1D mapping with safe output-size handling and proper CUDA-side checks, minimizing scaffold churn after prior anchor-marker drift issues. |
| `n5` | `strategy_07` | `True` | `True` | `1.4531043593130777` | `None` | `None` | Reduce branch divergence in the custom CUDA max-pooling kernel by precomputing valid kernel-loop bounds for each output element, so the inner reduction loop only iterates over in-bounds window positions. |
| `n6` | `strategy_06` | `False` | `True` | `None` | `builtins.RuntimeError` | `runtime` | Refine the existing custom CUDA max-pool path by focusing on the kernel launch/indexing region: keep a flattened 1D launch but reduce per-thread overhead via concrete launch tuning and output-size safety checks in the CUDA wrapper, while leaving the Python scaffold and fallbac... |
| `n7` | `strategy_04` | `False` | `True` | `None` | `builtins.RuntimeError` | `runtime` | Implement a simple custom CUDA max-pool forward kernel with one thread per output element, keeping the fast path limited to contiguous CUDA float32 NCHW tensors and preserving PyTorch fallback for all other cases. |
| `n8` | `strategy_04` | `True` | `True` | `0.9821428571428571` | `None` | `None` | Implement a simple custom CUDA max-pool forward kernel with one thread per output element, keeping the fast path limited to contiguous CUDA float32 NCHW tensors and preserving PyTorch fallback for all other cases. |
| `n9` | `strategy_04` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Use a simple custom CUDA max-pool forward kernel with one thread per output element and read-only cached loads, keeping a safe PyTorch fallback for unsupported cases. This targets the reduction directly while preserving MaxPool2d semantics. |
| `n10` | `strategy_03` | `False` | `False` | `None` | `SyntaxError` | `compile` | Specialize the CUDA max-pool forward path for the common square-kernel scalar-parameter case by storing scalar pooling parameters in init, adding a fixed 2x2/stride2/pad0/dilation1 fast kernel plus a generic fallback in cuda_cu, and selecting the specialized entrypoint from fo... |

## Failure Breakdown

- Stats: `{"attempt_count": 10, "debug_attempts": 3, "failure_counts": {"SyntaxError": 1, "anchor_marker_drift": 1, "broken_anchor_markers": 2, "builtins.RuntimeError": 4}, "failure_stage_counts": {"compile": 3, "runtime": 4}, "invalid_proposals": 0, "plan_attempts": 7, "pruned_count": 2}`
- Failure counts from nodes: `{'broken_anchor_markers': 2, 'anchor_marker_drift': 1, 'builtins.RuntimeError': 4, 'SyntaxError': 1}`
- Stage counts from nodes: `{'compile': 3, 'runtime': 4}`

Recent node log snippets:
- `n1` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n2` `anchor_marker_drift`: anchor_marker_drift: expected=[]; observed=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']
- `n3` `builtins.RuntimeError`: runtime_error=Error building extension 'stark_cuda_l1_p42_5e06aaea18e3': [1/3] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=stark_cu...
- `n3` `builtins.RuntimeError`: runtime_error_traceback=ninja: build stopped: subcommand failed.
- `n4` `builtins.RuntimeError`: runtime_error=Error building extension 'stark_cuda_l1_p42_be80a9ef1d68': [1/3] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=stark_cu...
- `n4` `builtins.RuntimeError`: runtime_error_traceback=ninja: build stopped: subcommand failed.
- `n6` `builtins.RuntimeError`: runtime_error=Error building extension 'stark_cuda_l1_p42_87c3fc7f8843': [1/3] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=stark_cu...
- `n6` `builtins.RuntimeError`: runtime_error_traceback=ninja: build stopped: subcommand failed.
- `n7` `builtins.RuntimeError`: runtime_error=Error building extension 'stark_cuda_l1_p42_dff50b8ebb97': [1/3] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=stark_cu...
- `n7` `builtins.RuntimeError`: runtime_error_traceback=ninja: build stopped: subcommand failed.
- `n9` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n10` `SyntaxError`: compilation_error_name=SyntaxError
- `n10` `SyntaxError`: compilation_error=Syntax error in custom generated code or ModelNew not found

## Code Artifacts

- run.json: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447/L1_P42_MaxPool2d_l1_p42/run.json`
- best_code.py: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447/L1_P42_MaxPool2d_l1_p42/best_code.py`
- best_node_id: `n2`
- best plan: Add a direct CUDA max-pool forward path for contiguous CUDA NCHW tensors, with one thread computing one output element and reducing the pooling window in registers, while keeping the existing PyTorch MaxPool2d fallback for unsupported cases.
- best strategy: `strategy_01`
