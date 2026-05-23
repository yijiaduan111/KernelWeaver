# Run Summary

- Run dir: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447`
- Tasks with run.json: `2`
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
| `L1_P10_TensorMatmul3D_l1_p10` | `1.001002004008016` | `root` | `10` | `8` | `4` | `10` | `matmul` |
| `L1_P1_SquareMatmul_l1_p1` | `1.0042016806722691` | `n8` | `10` | `5` | `4` | `10` | `matmul` |

# L1_P10_TensorMatmul3D_l1_p10

- Task name: `kernelbench_l1_10_10_3d_tensor_matrix_multiplication`
- Level/problem: `1/10`
- Backend: `cuda`
- Best node: `root`
- Speedup: `1.001002004008016`
- Source origin: `KernelBench/KernelBench/level1/10_3D_tensor_matrix_multiplication.py`

## Semantic Profile

- Enabled: `True`
- Mode: `rule`
- Op type: `matmul`
- Summary: Matrix multiplication style computation; preserve dimensions and consider tiled implementations.
- Recommended anchors: `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]`
- Risk notes: `["preserve M/N/K dimensions and transpose semantics", "handle tensor contiguity assumptions"]`

| # | Intent | Priority | Target Anchors | Summary |
|---:|---|---:|---|---|
| 1 | `use_tiled_matrix_multiply` | `5` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]` | use tiled matrix multiply |
| 2 | `preserve_matrix_shapes` | `4` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]` | preserve matrix shapes |

Anchor hints:
- `forward_stmt_1`: {"anchor_name": "forward_stmt_1", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site", "use tiled GEMM structure", "preserve strides and matrix dimensions"], "op_names": ["_3d_tensor_matrix_multiplication", "a", "along", "and", "args", "b", "d", "dimension", "from", "input", "k", "kernelbench"], "optimization_intents": ["use_tiled_matrix_multiply", "preserve_matrix_shapes"], "priority": 5, "region_role": "forward", "risk_notes": ["preserve M/N/K...
- `forward_stmt_2`: {"anchor_name": "forward_stmt_2", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site", "use tiled GEMM structure", "preserve strides and matrix dimensions"], "op_names": ["_3d_tensor_matrix_multiplication", "a", "along", "and", "args", "b", "d", "dimension", "from", "input", "k", "kernelbench"], "optimization_intents": ["use_tiled_matrix_multiply", "preserve_matrix_shapes"], "priority": 5, "region_role": "forward", "risk_notes": ["preserve M/N/K...
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
- Strategy count: `10`

| ID | Intent | Source | Scores | Anchors | Summary |
|---|---|---|---|---|---|
| `strategy_01` | `use_tiled_matrix_multiply` | `["openai-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 3.0, "openai-compatible": 5.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a custom CUDA path that treats the 3D tensor-matrix multiply as a batched GEMM over flattened (N*M) rows by K, using shared-memory tiling in K and L to improve global memory reuse versus a naive elementwise matmul path. |
| `strategy_02` | `Leverage cuBLAS directly via cublasGemmStridedBatchedEx for batched 3D matmul` | `["claude-compatible"]` | `{"claude-compatible": 5.0, "gemini-compatible": 2.0, "openai-compatible": 5.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Use cublasGemmStridedBatchedEx with batch_count=N, treating A as (N, M, K) with stride M*K and B as (K, L) with stride 0 (broadcast). This is the most hardware-optimal path for large batched GEMMs on NVIDIA GPUs. |
| `strategy_03` | `use_tiled_matrix_multiply` | `["gemini-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 5.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2"]` | Implement an advanced GEMM kernel with thread coarsening (register tiling) on the flattened (N*M, K) x (K, L) problem. |
| `strategy_08` | `use_tiled_matrix_multiply` | `["gemini-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 5.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2"]` | Flatten the 3D tensor A to 2D (N*M, K) and perform a standard tiled matrix multiplication with B (K, L) using shared memory. |
| `strategy_05` | `Custom tiled CUDA kernel with shared memory for 3D tensor-matrix multiply` | `["claude-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 3.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a tiled GEMM kernel that maps (batch, tile_m, tile_l) to CUDA blocks. Each block loads TILE_K-wide strips of A and B into shared memory, accumulates partial sums, and writes output. This exploits shared memory reuse across the K dimension. |
| `strategy_06` | `use_cublas` | `["gemini-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 2.0, "openai-compatible": 5.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Leverage cuBLAS SGEMM by treating the operation as a single matrix multiplication of (N*M, K) and (K, L). |
| `strategy_07` | `use_tiled_matrix_multiply` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 2.0, "openai-compatible": 4.0}` | `["cuda_cu", "forward_stmt_1", "forward_stmt_2"]` | Tune the CUDA kernel specifically for the broadcasted reuse pattern of B across the batch dimension, emphasizing cache/shared-memory reuse of B tiles across many A rows to exploit that the same matrix multiplies every (N,M) slice row. |
| `strategy_04` | `preserve_matrix_shapes` | `["openai-compatible"]` | `{"claude-compatible": 2.0, "gemini-compatible": 2.0, "openai-compatible": 4.0}` | `["forward_stmt_1", "forward_stmt_2", "cuda_cpp", "helpers"]` | Add a shape-specialized dispatch layer so the CUDA backend only takes over when the operation matches the exact 3D x 2D matmul pattern and favorable layout assumptions, while leaving unsupported or irregular cases on the PyTorch fallback path. |
| `strategy_09` | `Use cuBLAS batched GEMM via torch.matmul with contiguous tensors and explicit CUDA stream` | `["claude-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 2.0, "openai-compatible": 2.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Reshape A from (N,M,K) to (N*M, K) and B as (K,L), call cublasSgemm or use ATen's mm, then reshape output. This avoids Python overhead and ensures cuBLAS picks the optimal GEMM path. Expose via a custom CUDA extension that calls at::mm or at::matmul on contiguous views. |
| `strategy_10` | `use_tiled_matrix_multiply` | `["openai-compatible"]` | `{"claude-compatible": 2.0, "gemini-compatible": 2.0, "openai-compatible": 3.0}` | `["cuda_cpp", "cuda_cu", "forward_stmt_2"]` | Introduce a lightweight specialization strategy in the CUDA wrapper for common alignment-friendly sizes so the same tiled kernel family can use different launch parameters depending on K/L ranges, improving throughput without changing semantics. |

Implementation hints and risks:
- `strategy_01` hints: ["In cuda_cu, map A(N,M,K) x B(K,L) to output(N,M,L) with each block computing a tile of the flattened row dimension (N*M) and output columns L.", "Use a standard tiled GEMM structure with shared-memory staging for A and B tiles; preserve original logical indexing so output reshapes back to (N,M,L) without changing ModelNew I/O.", "In cuda_cpp, expose a single entrypoint that validates CUDA device, dtype, dimensionality, and output shape expectations before dispatch.", "In forward anchors, keep the torch.matmul fallback but route to the extension for supported CUDA-contiguous cases."]
- `strategy_01` risks: ["Preserve M/N/K dimensions and transpose semantics exactly; B is (K,L), not transposed.", "Handle non-contiguous tensors carefully, either by rejecting or materializing contiguous inputs before kernel launch.", "Boundary masking is required when flattened N*M or L are not multiples of tile sizes."]
- `strategy_02` hints: ["In cuda_cu, include <cublas_v2.h> and create/cache a cublasHandle_t", "Call cublasGemmStridedBatchedEx with: transa=CUBLAS_OP_N, transb=CUBLAS_OP_N, m=L, n=M, k=K, strideA=M*K, strideB=0 (same B for all batches), strideC=M*L, batchCount=N", "Use CUDA_R_32F for float32 inputs", "Cache the cublas handle in a static variable to avoid recreation overhead", "Ensure A and B are contiguous and on CUDA before calling"]
- `strategy_02` risks: ["strideB=0 may not be supported in all cuBLAS versions; verify API support", "Handle creation and caching needs thread safety consideration", "Must link against cublas library in extra_ldflags"]
- `strategy_03` hints: ["Use shared memory for block-level tiling (e.g., 128x128 tiles).", "Use register arrays for thread-level tiling (e.g., each thread computes an 8x8 sub-tile).", "Optimize memory access with float4 vectorized loads for both global and shared memory.", "Accumulate results in local registers before writing back to global memory."]
- `strategy_03` risks: ["High complexity in kernel implementation and index math.", "Increased register pressure may limit warp occupancy, requiring careful tuning of tile sizes."]
- `strategy_08` hints: ["Conceptually reshape A to (N*M, K) and C to (N*M, L) within the kernel.", "Use a 2D grid of thread blocks to compute tiles of the output.", "Load tiles of A and B into shared memory, synchronize threads, and compute partial dot products to reduce global memory bandwidth."]
- `strategy_08` risks: ["Ensure correct handling of boundary conditions if N*M or L is not a multiple of the tile size.", "Assumes tensors are contiguous in memory."]
- `strategy_05` hints: ["Define TILE_M=16, TILE_L=16, TILE_K=16 as compile-time constants", "Grid: (ceil(L/TILE_L), ceil(M/TILE_M), N) blocks; threads: (TILE_L, TILE_M)", "Each block loads A[n, m_tile:m_tile+TILE_M, k_tile:k_tile+TILE_K] and B[k_tile:k_tile+TILE_K, l_tile:l_tile+TILE_L] into shared memory", "Accumulate dot products over K in a loop with __syncthreads() between loads", "Handle boundary conditions with if-guards for non-multiple dimensions", "Use float4 vectorized loads where K stride allows"]
- `strategy_05` risks: ["Shared memory bank conflicts if tile dimensions not chosen carefully", "Boundary handling adds branch divergence for non-tile-multiple sizes", "May underperform cuBLAS for large regular sizes"]
- `strategy_06` hints: ["Include `<cublas_v2.h>` in the CUDA extension.", "Use `cublasSgemm` with `m=L`, `n=N*M`, `k=K`.", "Account for PyTorch's row-major layout by computing `B^T * A^T = C^T` in column-major cuBLAS semantics (i.e., pass B as the first matrix and A as the second)."]
- `strategy_06` risks: ["Requires linking against cuBLAS in the extension setup.", "Overhead of creating/destroying cuBLAS handles if not properly managed or cached."]
- `strategy_07` hints: ["Design block work so many output rows from flattened N*M share the same staged B tile, since B is constant across the batch.", "Prefer threadblock shapes that increase reuse of B along KxL tiles while keeping occupancy acceptable.", "If multiple dtypes are considered, start with a single common dtype path rather than overgeneralizing.", "Dispatch to this kernel only when tensor shapes are large enough that launch/setup overhead is amortized."]
- `strategy_07` risks: ["Aggressive shared-memory use can reduce occupancy if tile sizes are too large.", "Need to preserve exact indexing from flattened rows back to (N,M).", "Benefits may be small for tiny tensors where torch.matmul is already efficient."]
- `strategy_04` hints: ["In forward anchors, guard the custom path with checks for A.dim()==3, B.dim()==2, CUDA placement, compatible K dimension, and supported dtype/layout.", "Keep ModelNew I/O unchanged and preserve return shape (N,M,L) exactly.", "In cuda_cpp, centralize runtime validation and emit clear errors only for programmer misuse; otherwise prefer falling back in Python when unsupported.", "Use helpers only to maintain stable extension loading and avoid unnecessary recompilation churn."]
- `strategy_04` risks: ["Do not accidentally change broadcasting behavior expectations compared with torch.matmul for this exact signature.", "Contiguity assumptions must be explicit; a wrong fast path can silently corrupt indexing.", "Overly strict guards may reduce optimization coverage."]
- `strategy_09` hints: ["In cuda_cu, implement a wrapper that takes A (N,M,K) and B (K,L), calls A.contiguous().view({N*M, K}), then at::mm(A_2d, B), then view({N,M,L})", "Export the function via pybind11 in cuda_cpp as 'tensor_matmul'", "In forward_stmt_2, call _stark_get_extension().tensor_matmul(A, B) instead of torch.matmul", "Ensure both tensors are contiguous before the view operation", "Use TORCH_CHECK to validate shapes"]
- `strategy_09` risks: ["Must ensure A and B are contiguous; add .contiguous() calls", "View semantics require contiguous memory layout", "Shape validation needed to avoid silent errors"]
- `strategy_10` hints: ["In cuda_cpp, dispatch among a small number of kernel configurations based on K and L size buckets rather than one universal launch shape.", "Keep the number of variants small to avoid complexity; e.g., one config for small L, one for larger square-ish outputs.", "Use the forward call site only to invoke the extension entrypoint; keep shape logic mostly in the wrapper for maintainability.", "Retain the torch.matmul fallback for unsupported dtypes or pathological shapes."]
- `strategy_10` risks: ["Too many specializations increase maintenance burden and compile time.", "Incorrect dispatch thresholds can hurt performance instead of helping.", "Must preserve exact output dimensions and not assume divisibility by tile sizes."]

## Strategy Usage By Attempt

| Node | Strategy | Correct | Compile | Speedup | Failure | Stage | Summary |
|---|---|---:|---:|---:|---|---|---|
| `n1` | `strategy_01` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a custom CUDA tiled GEMM path by flattening A from (N,M,K) to (N*M,K), multiplying by B (K,L), and reshaping back to (N,M,L). Wire the forward path to use the extension for supported CUDA-contiguous float cases while preserving the torch.matmul fallback. |
| `n2` | `strategy_01` | `True` | `True` | `1.0` | `anchor_marker_drift` | `None` | Implement a custom CUDA tiled GEMM path by flattening A from (N,M,K) to (N*M,K), multiplying by B (K,L), and reshaping back to (N,M,L). Wire the forward path to use the extension for supported CUDA-contiguous float cases while preserving the torch.matmul fallback. |
| `n3` | `strategy_06` | `False` | `True` | `None` | `builtins.RuntimeError` | `runtime` | Use cuBLAS SGEMM on a flattened 2D view of the 3D tensor so the operation runs as a single vendor-optimized GEMM, while preserving the (N, M, L) output shape through reshape/view logic in the CUDA extension and guarded dispatch in forward. |
| `n4` | `strategy_02` | `False` | `False` | `None` | `SyntaxError` | `compile` | Use cuBLAS strided batched GEMM for the exact 3D tensor (N,M,K) times matrix (K,L) case, dispatching from forward only for supported CUDA float32 contiguous inputs and otherwise preserving the torch.matmul fallback. |
| `n5` | `strategy_06` | `True` | `True` | `1.0` | `None` | `None` | Use cuBLAS SGEMM on a flattened 2D view of the 3D tensor so the operation runs as a single vendor-optimized GEMM, while preserving the (N, M, L) output shape through reshape/view logic in the CUDA extension and guarded dispatch in forward. |
| `n6` | `strategy_02` | `False` | `True` | `None` | `builtins.RuntimeError` | `runtime` | Implement a cuBLAS strided-batched GEMM backend for the exact (N,M,K) x (K,L) case, using the same B for every batch and dispatching from forward only for supported CUDA-contiguous float32 inputs; otherwise keep the PyTorch fallback. |
| `n7` | `strategy_09` | `True` | `True` | `1.0` | `None` | `None` | Use the CUDA extension as a thin wrapper around ATen/cuBLAS by flattening A from (N,M,K) to (N*M,K), calling at::mm on contiguous CUDA tensors, and reshaping back to (N,M,L). This preserves semantics while avoiding handwritten kernel complexity. |
| `n8` | `strategy_02` | `False` | `True` | `None` | `builtins.RuntimeError` | `runtime` | Replace the placeholder extension with a real cuBLAS strided-batched GEMM backend for A(N,M,K) x B(K,L) -> C(N,M,L), and route the forward path to it only for supported CUDA float32 contiguous cases while keeping torch.matmul as fallback. |
| `n9` | `strategy_05` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a custom tiled CUDA kernel with shared memory for the exact 3D tensor (N,M,K) times 2D matrix (K,L) pattern, and dispatch to it from forward only for supported CUDA float32 contiguous inputs while preserving torch.matmul fallback otherwise. |
| `n10` | `strategy_04` | `False` | `True` | `None` | `builtins.AttributeError` | `runtime` | Add a robust shape-specialized dispatch layer in the forward path so the model only uses a CUDA extension for the exact supported 3D x 2D matmul case, while preserving a safe torch.matmul fallback for everything else. |

## Failure Breakdown

- Stats: `{"attempt_count": 10, "debug_attempts": 2, "failure_counts": {"SyntaxError": 1, "anchor_marker_drift": 1, "broken_anchor_markers": 2, "builtins.AttributeError": 1, "builtins.RuntimeError": 3}, "failure_stage_counts": {"compile": 3, "runtime": 4}, "invalid_proposals": 0, "plan_attempts": 8, "pruned_count": 0}`
- Failure counts from nodes: `{'broken_anchor_markers': 2, 'anchor_marker_drift': 1, 'builtins.RuntimeError': 3, 'SyntaxError': 1, 'builtins.AttributeError': 1}`
- Stage counts from nodes: `{'compile': 3, 'runtime': 4}`

Recent node log snippets:
- `n1` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n2` `anchor_marker_drift`: anchor_marker_drift: expected=[]; observed=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']
- `n3` `builtins.RuntimeError`: runtime_error=Error building extension 'stark_cuda_l1_p10_74b04aacc9ac': [1/3] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=stark_cuda_l1_p10_74b04aacc9ac -DTORCH_API_INCLUDE_EXTENSION_H -isystem /data/dyj/minic...
- `n3` `builtins.RuntimeError`: runtime_error_traceback=ninja: build stopped: subcommand failed.
- `n4` `SyntaxError`: compilation_error_name=SyntaxError
- `n4` `SyntaxError`: compilation_error=Syntax error in custom generated code or ModelNew not found
- `n6` `builtins.RuntimeError`: runtime_error=Error building extension 'stark_cuda_l1_p10_4f8a33032d1c': [1/3] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=stark_cu...
- `n6` `builtins.RuntimeError`: runtime_error_traceback=ninja: build stopped: subcommand failed.
- `n8` `builtins.RuntimeError`: runtime_error=Error building extension 'stark_cuda_l1_p10_e22c00c184b4': [1/3] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=stark_cuda_l1_p10_e22c00c184b4 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /data/dyj/minic...
- `n8` `builtins.RuntimeError`: runtime_error_traceback=ninja: build stopped: subcommand failed.
- `n9` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n10` `builtins.AttributeError`: runtime_error='NoneType' object has no attribute 'shape'
- `n10` `builtins.AttributeError`: runtime_error_traceback=AttributeError: 'NoneType' object has no attribute 'shape'

## Code Artifacts

- run.json: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447/L1_P10_TensorMatmul3D_l1_p10/run.json`
- best_code.py: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447/L1_P10_TensorMatmul3D_l1_p10/best_code.py`
- best_node_id: `root`
- best plan: -
- best strategy: `None`

# L1_P1_SquareMatmul_l1_p1

- Task name: `kernelbench_l1_1_1_square_matrix_multiplication`
- Level/problem: `1/1`
- Backend: `cuda`
- Best node: `n8`
- Speedup: `1.0042016806722691`
- Source origin: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`

## Semantic Profile

- Enabled: `True`
- Mode: `rule`
- Op type: `matmul`
- Summary: Matrix multiplication style computation; preserve dimensions and consider tiled implementations.
- Recommended anchors: `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]`
- Risk notes: `["preserve M/N/K dimensions and transpose semantics", "handle tensor contiguity assumptions"]`

| # | Intent | Priority | Target Anchors | Summary |
|---:|---|---:|---|---|
| 1 | `use_tiled_matrix_multiply` | `5` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]` | use tiled matrix multiply |
| 2 | `preserve_matrix_shapes` | `4` | `["forward_stmt_1", "forward_stmt_2", "cuda_cu", "cuda_cpp", "helpers"]` | preserve matrix shapes |

Anchor hints:
- `forward_stmt_1`: {"anchor_name": "forward_stmt_1", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site", "use tiled GEMM structure", "preserve strides and matrix dimensions"], "op_names": ["_square_matrix_multiplication_", "a", "args", "b", "c", "input", "kernelbench", "level1", "matmul", "matrix", "model", "multiplication"], "optimization_intents": ["use_tiled_matrix_multiply", "preserve_matrix_shapes"], "priority": 5, "region_role": "forward", "risk_notes": ["p...
- `forward_stmt_2`: {"anchor_name": "forward_stmt_2", "backend_hints": ["keep ModelNew I/O unchanged", "edit cuda_cu/cuda_cpp plus the forward call site", "use tiled GEMM structure", "preserve strides and matrix dimensions"], "op_names": ["_square_matrix_multiplication_", "a", "args", "b", "c", "input", "kernelbench", "level1", "matmul", "matrix", "model", "multiplication"], "optimization_intents": ["use_tiled_matrix_multiply", "preserve_matrix_shapes"], "priority": 5, "region_role": "forward", "risk_notes": ["p...
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
- Strategy count: `10`

| ID | Intent | Source | Scores | Anchors | Summary |
|---|---|---|---|---|---|
| `strategy_01` | `use_tiled_matrix_multiply` | `["openai-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 5.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a custom CUDA square-GEMM path using shared-memory tiling so each thread block computes a C tile while cooperatively loading A/B tiles, reducing global memory traffic versus a naive elementwise formulation. |
| `strategy_05` | `use_tiled_matrix_multiply_with_thread_coarsening` | `["gemini-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 5.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2"]` | Implement a tiled matrix multiplication with thread coarsening (register tiling). |
| `strategy_08` | `use_tiled_shared_memory_gemm_kernel` | `["claude-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a custom tiled GEMM CUDA kernel using shared memory tiles (e.g., 32x32 tiles) to maximize data reuse and reduce global memory bandwidth. This avoids library overhead and demonstrates explicit tiling for square matrices. |
| `strategy_02` | `use_cublas_for_matmul` | `["gemini-compatible"]` | `{"claude-compatible": 5.0, "gemini-compatible": 3.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2"]` | Use cuBLAS for optimal matrix multiplication performance. |
| `strategy_07` | `use_tiled_matrix_multiply` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 4.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp"]` | Tune the CUDA tiled GEMM for backend specifics by selecting conservative tile sizes, coalesced loads, and enough per-thread work to balance occupancy and arithmetic intensity on typical CUDA GPUs. |
| `strategy_04` | `use_cublas_gemm_via_custom_cuda_extension` | `["claude-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 3.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Replace torch.matmul with a custom CUDA extension that calls cuBLAS SGEMM directly, exposing it via pybind11. cuBLAS is highly optimized for square matrix multiplications and will outperform the generic torch.matmul dispatch overhead for fixed square sizes. |
| `strategy_03` | `preserve_matrix_shapes` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 2.0, "openai-compatible": 4.0}` | `["forward_stmt_1", "forward_stmt_2", "cuda_cpp", "helpers"]` | Add strict shape/stride/dtype gating around the custom kernel so the optimized path only handles the simple square dense case, while all other cases retain the baseline PyTorch matmul behavior. |
| `strategy_06` | `use_wmma_tensor_cores` | `["gemini-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 4.0, "openai-compatible": 2.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2"]` | Implement matrix multiplication using CUDA WMMA (Tensor Cores) API. |
| `strategy_10` | `use_tiled_matrix_multiply` | `["openai-compatible"]` | `{"claude-compatible": 2.0, "gemini-compatible": 2.0, "openai-compatible": 4.0}` | `["forward_stmt_1", "forward_stmt_2", "init_body"]` | Introduce a lightweight fast path policy: use the custom CUDA GEMM only above a matrix-size threshold and keep torch.matmul for tiny problems where library/kernel-launch overhead can be better. |
| `strategy_09` | `use_wmma_tensor_core_gemm` | `["claude-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 3.0, "openai-compatible": 1.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Leverage CUDA Tensor Cores via the WMMA API (wmma::fragment) for FP16 or mixed-precision matrix multiplication, achieving significantly higher throughput on Volta+ GPUs. Convert FP32 inputs to FP16, compute with Tensor Cores, accumulate in FP32. |

Implementation hints and risks:
- `strategy_01` hints: ["Keep ModelNew inputs/outputs unchanged and dispatch to the extension only for CUDA tensors with supported dtype/layout; otherwise fall back to torch.matmul.", "In cuda_cu, use a standard block_m/block_n/block_k tiled GEMM structure with shared-memory staging and boundary guards, preserving M/N/K interpretation for square matrices.", "In cuda_cpp, expose one thin entrypoint that validates device, dimensionality, square shape compatibility, and returns a freshly allocated output tensor.", "At the forward call site, call _stark_get_extension() lazily and route A,B through the custom entrypoint without changing Python-visible semantics."]
- `strategy_01` risks: ["Preserve M/N/K dimensions and transpose semantics exactly as torch.matmul for 2D inputs.", "Handle tensor contiguity assumptions explicitly or force a safe fallback for non-contiguous inputs.", "Performance can regress for very small matrices due to extension and kernel launch overhead."]
- `strategy_05` hints: ["Use shared memory for block-level tiling and registers for thread-level tiling.", "Assign each thread to compute a small 2D tile (e.g., 4x4 or 8x8) of the output matrix rather than a single element.", "Use float4 vector loads for global memory reads to maximize memory bandwidth utilization.", "Export and call the kernel in forward_stmt_2."]
- `strategy_05` risks: ["Increased register pressure may reduce warp occupancy.", "Complex boundary handling if matrix size is not a multiple of the block/tile size."]
- `strategy_08` hints: ["In cuda_cu, write a __global__ kernel with TILE_SIZE=32; each thread block loads a 32x32 tile of A and B into __shared__ memory, accumulates partial dot products in a register, then writes to C", "Use __syncthreads() between load and compute phases within each tile iteration", "Launch with dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE) and dim3 block(TILE_SIZE, TILE_SIZE)", "Handle boundary conditions with if-guards when N is not a multiple of TILE_SIZE", "In cuda_cpp, export the launcher as 'tiled_matmul'", "In forward_stmt_2, replace torch.matmul with _stark_get_extension().tiled_matmul(A.contiguous(), B.contiguous(), N)", "Add '--ptxas-options=-v' to extra_cuda_cflag...
- `strategy_08` risks: ["Shared memory bank conflicts can degrade performance if tile layout is not carefully chosen; consider padding shared memory arrays by 1 column", "Register pressure with TILE_SIZE=32 may cause spilling; consider TILE_SIZE=16 as a safer alternative", "Boundary handling adds branch divergence for non-power-of-2 matrix sizes", "This approach is unlikely to beat cuBLAS for large matrices"]
- `strategy_02` hints: ["Include <ATen/cuda/CUDAContext.h> and use at::cuda::blas::gemm or direct cuBLAS API calls.", "Set up the correct leading dimensions (lda, ldb, ldc) and transpose flags. Note that cuBLAS is column-major, so computing A*B in row-major is equivalent to B*A in column-major.", "Export the function via pybind11 and call it in forward_stmt_2."]
- `strategy_02` risks: ["Requires careful handling of row-major to column-major translation.", "May bypass the intent of writing a custom CUDA kernel if the benchmark strictly evaluates handwritten CUDA code."]
- `strategy_07` hints: ["Prefer simple tile shapes that map well to warp memory access patterns and avoid excessive shared-memory or register pressure.", "Arrange global memory reads so adjacent threads load adjacent elements from A and B wherever layout permits, then accumulate multiple products in registers before writing C once.", "Keep wrapper/export structure minimal in cuda_cpp so host overhead stays small relative to the kernel work.", "Use the existing build flags in helpers as-is and rely on CUDA-side organization rather than extra compilation complexity."]
- `strategy_07` risks: ["Aggressive tile sizing can lower occupancy or exceed shared-memory limits on some devices.", "Backend-specific tuning may not generalize equally across all GPU architectures."]
- `strategy_04` hints: ["In cuda_cu, include <cublas_v2.h> and implement a wrapper function that obtains a cuBLAS handle (cached as a static), calls cublasSgemm with alpha=1.0 and beta=0.0, using CUBLAS_OP_N for both A and B (note column-major vs row-major: pass B first then A to get C=A*B in row-major)", "Allocate output tensor with torch::zeros({N, N}, A.options()) before calling cublasSgemm", "In cuda_cpp, export the wrapper function via PYBIND11_MODULE as 'square_matmul'", "In forward_stmt_1/forward_stmt_2, call _stark_get_extension().square_matmul(A.contiguous(), B.contiguous()) instead of torch.matmul", "Add '-lcublas' to extra_cuda_cflags or link via extra_ldflags in the helpers load_inline call", "Cache...
- `strategy_04` risks: ["cuBLAS uses column-major order; must swap A and B arguments to cublasSgemm to get correct row-major C=A*B result", "Must ensure tensors are contiguous before passing pointers", "cuBLAS handle lifecycle must be managed carefully to avoid leaks", "Link flags must be set correctly for cublas"]
- `strategy_03` hints: ["Use helpers/forward anchors to keep lazy extension loading isolated and avoid changing module construction behavior.", "In the forward path, guard on CUDA device, 2D tensors, matching inner dimensions, square expectations if required by the benchmark, and supported dtypes.", "In cuda_cpp, mirror these checks at the C++ boundary so invalid calls fail clearly instead of producing silent wrong answers.", "Retain torch.matmul as the fallback to preserve exact semantics for unsupported layouts or edge cases."]
- `strategy_03` risks: ["Overly strict gating may reduce optimization coverage and hide performance gains.", "Need consistent shape policy between Python and C++ wrappers to avoid mismatched behavior."]
- `strategy_06` hints: ["Include <mma.h> and use the nvcuda::wmma namespace.", "Declare wmma::fragment for matrices A, B, and accumulator C.", "Load data from shared memory into fragments, perform wmma::mma_sync, and store the result back.", "Use TF32 mode if inputs are FP32 and running on Ampere+ architectures, or cast to FP16/BF16."]
- `strategy_06` risks: ["Hardware dependent (requires Volta architecture or newer).", "Data types must be compatible with Tensor Core requirements, which may require precision casting or TF32 enablement."]
- `strategy_10` hints: ["Store a simple threshold or policy knob in init_body without changing external API behavior.", "In forward, branch on N and device so tiny square matrices continue using torch.matmul while larger ones use the extension path.", "Keep the threshold conservative and benchmark-driven for this level-1 workload to avoid overfitting to one exact size.", "Maintain identical returned shape and dtype on both branches."]
- `strategy_10` risks: ["Threshold choice is hardware-dependent and may need retuning.", "Branching policy adds complexity and can reduce reproducibility if not documented clearly."]
- `strategy_09` hints: ["In cuda_cu, include <mma.h> and use nvcuda::wmma namespace; define WMMA_M=16, WMMA_N=16, WMMA_K=16 fragments", "Convert input tensors to half precision using A.to(torch::kFloat16) before kernel launch, or handle conversion inside the kernel", "Each warp handles a 16x16 output tile; use wmma::load_matrix_sync, wmma::mma_sync, wmma::store_matrix_sync", "Accumulate in float fragments (ACCUMULATOR type) for numerical accuracy, then store back as float32", "Pad matrix dimensions to multiples of 16 if needed", "In helpers, add '-arch=sm_70' or higher to extra_cuda_cflags to enable Tensor Core instructions", "Export as 'wmma_matmul' in cuda_cpp and call from forward_stmt_2"]
- `strategy_09` risks: ["Requires sm_70+ (Volta or newer); will fail to compile on older GPUs", "FP16 accumulation may introduce numerical precision differences vs reference torch.matmul in FP32", "WMMA requires matrix dimensions to be multiples of 16; padding logic adds complexity", "The conversion overhead from FP32 to FP16 may negate gains for small matrices", "Correctness validation against torch.matmul FP32 output requires tolerance adjustment"]

## Strategy Usage By Attempt

| Node | Strategy | Correct | Compile | Speedup | Failure | Stage | Summary |
|---|---|---:|---:|---:|---|---|---|
| `n1` | `strategy_01` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a custom CUDA square-GEMM path using shared-memory tiling, with a thin C++ wrapper and a guarded Python forward dispatch that preserves fallback semantics. |
| `n2` | `strategy_01` | `True` | `True` | `0.12656250000000002` | `anchor_marker_drift` | `None` | Implement a custom CUDA square-GEMM path using shared-memory tiling, with a thin C++ wrapper and a guarded Python forward dispatch that preserves fallback semantics. |
| `n3` | `strategy_03` | `False` | `False` | `None` | `SyntaxError` | `compile` | Add strict fast-path gating so the custom CUDA kernel is only invoked for the exact dense square CUDA case, while all other inputs preserve baseline torch.matmul semantics. This directly addresses correctness/stability concerns and avoids expensive extension dispatch on unsupp... |
| `n4` | `strategy_08` | `False` | `False` | `None` | `SyntaxError` | `compile` | Implement a conservative shared-memory tiled CUDA GEMM extension and route the forward matmul through it only for contiguous CUDA 2D square float32 inputs, otherwise keep torch.matmul fallback. |
| `n5` | `strategy_03` | `True` | `True` | `1.0` | `None` | `None` | Add strict fast-path gating so the custom CUDA kernel is only invoked for the exact dense square CUDA case, while all other inputs preserve baseline torch.matmul semantics. This directly addresses correctness/stability concerns and avoids expensive extension dispatch on unsupp... |
| `n6` | `strategy_01` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a custom CUDA square-GEMM path using a shared-memory tiled kernel, expose it through the C++/pybind extension, and dispatch to it from forward only for supported CUDA dense 2D inputs while preserving torch.matmul fallback semantics. |
| `n7` | `strategy_03` | `False` | `True` | `None` | `builtins.AttributeError` | `runtime` | Add strict shape/device/dtype/layout gating around the matmul call so any future/custom extension path only runs for the simple supported square dense CUDA case, while all other inputs retain baseline torch.matmul semantics. |
| `n8` | `strategy_03` | `True` | `True` | `1.0042016806722691` | `anchor_marker_drift` | `None` | Add strict shape/device/dtype/layout gating around the matmul call so any future/custom extension path only runs for the simple supported square dense CUDA case, while all other inputs retain baseline torch.matmul semantics. |
| `n9` | `strategy_01` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a custom CUDA square-GEMM path using shared-memory tiling, with a thin C++ wrapper and a guarded Python forward fast path that preserves the ModelNew interface and falls back to torch.matmul when unsupported. |
| `n10` | `strategy_08` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a conservative shared-memory tiled CUDA GEMM and wire forward to use it only for contiguous CUDA 2D square float32 inputs, otherwise fall back to torch.matmul. This targets the benchmarked square-matmul case while minimizing scaffold churn and avoiding anchor-marker ... |

## Failure Breakdown

- Stats: `{"attempt_count": 10, "debug_attempts": 3, "failure_counts": {"SyntaxError": 2, "anchor_marker_drift": 2, "broken_anchor_markers": 4, "builtins.AttributeError": 1}, "failure_stage_counts": {"compile": 6, "runtime": 1}, "invalid_proposals": 0, "plan_attempts": 7, "pruned_count": 1}`
- Failure counts from nodes: `{'broken_anchor_markers': 4, 'anchor_marker_drift': 2, 'SyntaxError': 2, 'builtins.AttributeError': 1}`
- Stage counts from nodes: `{'compile': 6, 'runtime': 1}`

Recent node log snippets:
- `n1` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n2` `anchor_marker_drift`: anchor_marker_drift: expected=[]; observed=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']
- `n3` `SyntaxError`: compilation_error_name=SyntaxError
- `n3` `SyntaxError`: compilation_error=Syntax error in custom generated code or ModelNew not found
- `n4` `SyntaxError`: compilation_error_name=SyntaxError
- `n4` `SyntaxError`: compilation_error=Syntax error in custom generated code or ModelNew not found
- `n6` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n7` `builtins.AttributeError`: runtime_error='NoneType' object has no attribute 'shape'
- `n7` `builtins.AttributeError`: runtime_error_traceback=AttributeError: 'NoneType' object has no attribute 'shape'
- `n8` `anchor_marker_drift`: anchor_marker_drift: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']
- `n9` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n10` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]

## Code Artifacts

- run.json: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447/L1_P1_SquareMatmul_l1_p1/run.json`
- best_code.py: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447/L1_P1_SquareMatmul_l1_p1/best_code.py`
- best_node_id: `n8`
- best plan: Add strict shape/device/dtype/layout gating around the matmul call so any future/custom extension path only runs for the simple supported square dense CUDA case, while all other inputs retain baseline torch.matmul semantics.
- best strategy: `strategy_03`
