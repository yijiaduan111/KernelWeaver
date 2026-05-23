# Run Summary

- Run dir: `/data/dyj/KernelWeaver/runs/p1_single_cuda_cudallm_delib_main_20260513_124845`
- Tasks with run.json: `1`
- Summary rows: `1`
- task_count: `1`
- success_count: `1`
- compile_rate: `0.4`
- correct_rate: `0.4`
- best_speedup: `0.9831932773109245`
- median_speedup: `0.9831932773109245`
- improved_over_reference_rate: `0.0`
- paper_metrics: `{"Fast1": 0.0, "Speed": 0.9831932773109245, "Success": 1.0}`

| Task | Speedup | Best | Attempts | Compile OK | Correct | Strategies | Op Type |
|---|---:|---|---:|---:|---:|---:|---|
| `L1_P1_SquareMatmul_l1_p1` | `0.9831932773109245` | `root` | `10` | `5` | `5` | `10` | `matmul` |

# L1_P1_SquareMatmul_l1_p1

- Task name: `kernelbench_l1_1_1_square_matrix_multiplication`
- Level/problem: `1/1`
- Backend: `cuda`
- Best node: `root`
- Speedup: `0.9831932773109245`
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
| `strategy_01` | `use_tiled_matrix_multiply` | `["openai-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 5.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a custom CUDA tiled GEMM path and dispatch to it from the forward call site, keeping the Python ModelNew interface unchanged while replacing the generic matmul path for square CUDA inputs. |
| `strategy_02` | `use_register_tiling` | `["gemini-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 5.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2", "helpers"]` | Optimize the tiled matrix multiplication by having each thread compute multiple output elements. |
| `strategy_04` | `use_tiled_shared_memory_gemm_kernel` | `["claude-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 4.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2"]` | Implement a custom tiled GEMM CUDA kernel using shared memory tiles (e.g., 32x32) with double-buffering to maximize arithmetic intensity and hide memory latency for square matrix multiplication. |
| `strategy_08` | `use_vectorized_memory_access_with_float4_tiled_kernel` | `["claude-compatible"]` | `{"claude-compatible": 4.0, "gemini-compatible": 5.0, "openai-compatible": 3.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2"]` | Enhance the tiled GEMM kernel with float4 vectorized loads to maximize memory bandwidth utilization, loading 4 floats per instruction and reducing load instruction count by 4x. |
| `strategy_07` | `preserve_matrix_shapes` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 3.0, "openai-compatible": 5.0}` | `["forward_stmt_1", "forward_stmt_2", "cuda_cpp"]` | Add a shape-aware dispatch layer that only routes exact supported square-matrix cases into the custom kernel and keeps all other cases on torch.matmul to avoid semantic regressions. |
| `strategy_05` | `use_tiled_matrix_multiply` | `["gemini-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 3.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2", "helpers"]` | Implement a basic shared memory tiled matrix multiplication. |
| `strategy_09` | `improve_memory_access_coalescing` | `["openai-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 3.0, "openai-compatible": 4.0}` | `["cuda_cu", "cuda_cpp"]` | Structure the CUDA kernel so global loads/stores are coalesced and shared-memory tile usage minimizes redundant reads, complementing the tiled GEMM strategy. |
| `strategy_06` | `use_vectorized_memory_access` | `["gemini-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 4.0, "openai-compatible": 2.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2", "helpers"]` | Use vectorized loads (e.g., float4) to improve global memory read bandwidth. |
| `strategy_03` | `use_tensor_cores` | `["gemini-compatible"]` | `{"claude-compatible": 3.0, "gemini-compatible": 3.0, "openai-compatible": 1.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_2", "helpers"]` | Utilize CUDA WMMA (Warp Matrix Multiply Accumulate) API for hardware-accelerated matrix multiplication. |
| `strategy_10` | `use_wmma_tensor_core_tiled_gemm` | `["claude-compatible"]` | `{"claude-compatible": 2.0, "gemini-compatible": 3.0, "openai-compatible": 1.0}` | `["cuda_cu", "cuda_cpp", "forward_stmt_1", "forward_stmt_2", "helpers"]` | Implement a WMMA (Warp Matrix Multiply Accumulate) kernel using CUDA tensor cores for float16 or mixed-precision computation, achieving peak throughput on Volta+ GPUs. |

Implementation hints and risks:
- `strategy_01` hints: ["In cuda_cu, add a block-tiled GEMM kernel using shared memory for A/B tiles and per-thread accumulation for C.", "Use a simple square-friendly tile size choice and boundary guards so dimensions remain correct even when N is not an exact multiple of tile size.", "In cuda_cpp, expose a minimal entrypoint that validates CUDA device, dtype, dimensionality, and allocates the output tensor before launching the kernel.", "In the forward anchors, call _stark_get_extension().<entrypoint>(A, B) only for supported CUDA-contiguous cases, otherwise preserve the existing torch.matmul fallback."]
- `strategy_01` risks: ["Preserve M/N/K dimensions and transpose semantics exactly as torch.matmul for 2D square inputs.", "Handle tensor contiguity assumptions carefully or explicitly fall back when unsupported.", "Tile size that is too large may hurt occupancy or shared-memory usage on smaller GPUs."]
- `strategy_02` hints: ["Use a 2D thread block but assign each thread to compute a 2x2, 4x4, or 8x8 block of the output matrix.", "Accumulate partial results in thread-local registers.", "Load data from shared memory to registers to maximize arithmetic intensity."]
- `strategy_02` risks: ["High register pressure may reduce warp occupancy.", "Boundary handling becomes significantly more complex."]
- `strategy_04` hints: ["In cuda_cu, write a `__global__ void tiled_matmul_kernel(float* A, float* B, float* C, int N)` with TILE_SIZE=32, using `__shared__ float As[32][32], Bs[32][32]`", "Each thread block computes a 32x32 tile of C; each thread computes one element by iterating over K-dimension tiles with `__syncthreads()` between loads and computes", "Use `__ldg()` for read-only global memory loads of A and B to leverage L1 texture cache", "Launch with `dim3 grid((N+31)/32, (N+31)/32)` and `dim3 block(32, 32)`", "Wrap in `torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B)` that allocates output with `torch::zeros({N, N}, A.options())` and launches the kernel", "Export via pybind11 in cuda_cpp and ...
- `strategy_04` risks: ["Only handles float32; add dtype checks or template the kernel for float16", "Boundary handling needed when N is not divisible by TILE_SIZE", "For very large N, cuBLAS with tensor cores will outperform a hand-written kernel"]
- `strategy_08` hints: ["In cuda_cu, use `float4` loads when reading rows of A and columns of B into shared memory: `reinterpret_cast<float4*>(As_row)[j/4] = reinterpret_cast<const float4*>(A_row)[j/4]`", "Pad shared memory arrays to avoid bank conflicts: declare `__shared__ float As[32][33]` (extra column breaks 32-way bank conflicts)", "Each thread loads a float4 (4 consecutive elements) per tile iteration, reducing global memory transactions", "Ensure N is padded to a multiple of 4 in the wrapper, or handle the remainder with scalar loads", "Keep TILE_SIZE=32 for the block dimension; inner loop unrolling with `#pragma unroll` over the tile K dimension", "Register-level accumulation: each thread maintains a l...
- `strategy_08` risks: ["Alignment requirements: input pointers must be 16-byte aligned for float4; torch tensors are typically 256-byte aligned so this is safe", "N must be divisible by 4 or boundary scalar fallback needed", "Shared memory bank conflict analysis must account for the access pattern of the specific tile layout"]
- `strategy_07` hints: ["In forward, gate the custom path on 2D tensors, matching inner dimensions, CUDA placement, and expected dtype/layout.", "Keep output shape identical to (A.rows, B.cols), even if the benchmark is square-focused; this avoids hard-coding assumptions into the public interface.", "In cuda_cpp, mirror lightweight checks so invalid launches are rejected early rather than producing silent wrong answers.", "Retain torch.matmul fallback for CPU, noncontiguous tensors, unsupported dtypes, or edge shapes."]
- `strategy_07` risks: ["Overly strict gating can reduce benchmark coverage and hide the optimized path.", "Do not accidentally enforce square-only semantics at the interface level if the baseline accepts more general 2D matmul."]
- `strategy_05` hints: ["Define a CUDA kernel with 2D thread blocks (e.g., 32x32).", "Load tiles of matrices A and B into shared memory.", "Use __syncthreads() to synchronize before and after computing the dot product for the tile.", "Export the kernel via pybind11 and call it in forward_stmt_2."]
- `strategy_05` risks: ["Must handle matrix dimensions that are not perfectly divisible by the tile size using boundary checks."]
- `strategy_09` hints: ["Map threads so neighboring lanes read neighboring elements from A/B tiles and write contiguous regions of C.", "Keep shared-memory layout simple and consider padding only if bank conflicts appear significant for the chosen tile geometry.", "Use contiguous-input assumptions explicitly in the kernel contract to preserve straightforward indexing and maximize coalescing.", "Launch with 2D thread blocks aligned to the tile geometry to match row/column traversal of the matrix product."]
- `strategy_09` risks: ["Indexing mistakes can silently transpose or scramble results; preserve row/column semantics carefully.", "Padding or more complex layouts may increase shared-memory footprint and reduce occupancy if overused."]
- `strategy_06` hints: ["Cast global memory pointers to float4* (or equivalent) for loading tiles of A and B into shared memory.", "Ensure shared memory is padded to avoid bank conflicts when writing vectorized loads.", "Process 4 elements per instruction."]
- `strategy_06` risks: ["Requires matrix dimensions and strides to be aligned to the vector size (e.g., multiples of 4).", "Fallback to scalar loads needed for unaligned edges."]
- `strategy_03` hints: ["Include <mma.h> and use nvcuda::wmma namespace.", "Load fragments of A and B from shared/global memory into WMMA registers.", "Use wmma::mma_sync to perform the matrix multiplication.", "Store the accumulator fragments back to global memory."]
- `strategy_03` risks: ["Requires specific data types (e.g., FP16, BF16, or TF32) and hardware support.", "Matrix dimensions must be multiples of the WMMA fragment sizes (e.g., 16x16x16)."]
- `strategy_10` hints: ["In cuda_cu, include `<mma.h>` and use `nvcuda::wmma` namespace with `wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>` fragments", "Use WMMA tile size of 16x16x16; each warp computes one 16x16 output tile by accumulating over K in steps of 16", "Convert input tensors to half precision inside the wrapper if they are float32, or require half input", "Use shared memory for staging A and B tiles before loading into WMMA fragments to coalesce global memory access", "In helpers, add `-arch=sm_70` or higher to extra_cuda_cflags to enable tensor core instructions", "Export wrapper `wmma_matmul(torch::Tensor A, torch::Tensor B)` via pybind11 in cuda_cpp"]
- `strategy_10` risks: ["Requires GPU with compute capability >= 7.0 (Volta+)", "WMMA requires matrix dimensions to be multiples of 16; need padding logic", "Precision loss from FP16 conversion may cause correctness issues if the benchmark checks against FP32 reference", "Complexity is high; cuBLAS with CUBLAS_COMPUTE_16F may be simpler and equally fast"]

## Strategy Usage By Attempt

| Node | Strategy | Correct | Compile | Speedup | Failure | Stage | Summary |
|---|---|---:|---:|---:|---|---|---|
| `n1` | `strategy_01` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a custom CUDA shared-memory tiled GEMM and dispatch to it only for supported CUDA float32 contiguous 2D inputs, while preserving torch.matmul as the fallback path. |
| `n2` | `strategy_01` | `True` | `True` | `0.12153846153846154` | `anchor_marker_drift` | `None` | Implement a custom CUDA shared-memory tiled GEMM and dispatch to it only for supported CUDA float32 contiguous 2D inputs, while preserving torch.matmul as the fallback path. |
| `n3` | `strategy_04` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Replace the current small-tile custom matmul with a more square-friendly shared-memory tiled GEMM kernel using a larger tile and explicit guarded dispatch from forward, while preserving the ModelNew interface and torch.matmul fallback for unsupported cases. |
| `n4` | `strategy_04` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a shared-memory tiled CUDA GEMM kernel and wire it through a guarded forward-path dispatch, preserving the ModelNew API and falling back to torch.matmul for unsupported cases. |
| `n5` | `strategy_04` | `True` | `True` | `0.125130890052356` | `anchor_marker_drift` | `None` | Replace the current small-tile custom matmul with a more square-friendly shared-memory tiled GEMM kernel using a larger tile and explicit guarded dispatch from forward, while preserving the ModelNew interface and torch.matmul fallback for unsupported cases. |
| `n6` | `strategy_01` | `True` | `True` | `0.12474226804123713` | `None` | `None` | Implement a custom CUDA tiled GEMM path and dispatch to it from the forward call site, while preserving the existing ModelNew interface and exact fallback behavior for unsupported cases. |
| `n7` | `strategy_07` | `False` | `False` | `None` | `SyntaxError` | `compile` | Add a shape- and layout-aware dispatch in the forward path that only routes supported 2D CUDA float32 contiguous matmul cases into a custom extension entrypoint, preserving exact fallback semantics for all other inputs. |
| `n8` | `strategy_07` | `True` | `True` | `0.9831932773109245` | `anchor_marker_drift` | `None` | Add a shape- and layout-aware dispatch in the forward path that only routes supported 2D CUDA float32 contiguous matmul cases into a custom extension entrypoint, preserving exact fallback semantics for all other inputs. |
| `n9` | `strategy_01` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a real custom CUDA tiled GEMM kernel and dispatch to it only for supported CUDA float32 contiguous 2D inputs, preserving the ModelNew interface and torch.matmul fallback for unsupported cases. |
| `n10` | `strategy_04` | `False` | `False` | `None` | `broken_anchor_markers` | `compile` | Implement a real shared-memory tiled GEMM CUDA kernel and keep the forward dispatch guarded so supported CUDA float32 contiguous 2D inputs use the custom path while everything else falls back to torch.matmul. |

## Failure Breakdown

- Stats: `{"attempt_count": 10, "debug_attempts": 3, "failure_counts": {"SyntaxError": 1, "anchor_marker_drift": 3, "broken_anchor_markers": 5}, "failure_stage_counts": {"compile": 6}, "invalid_proposals": 0, "plan_attempts": 7, "pruned_count": 1}`
- Failure counts from nodes: `{'broken_anchor_markers': 5, 'anchor_marker_drift': 3, 'SyntaxError': 1}`
- Stage counts from nodes: `{'compile': 6}`

Recent node log snippets:
- `n1` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n2` `anchor_marker_drift`: anchor_marker_drift: expected=[]; observed=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']
- `n3` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n4` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n5` `anchor_marker_drift`: anchor_marker_drift: expected=[]; observed=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']
- `n7` `SyntaxError`: compilation_error_name=SyntaxError
- `n7` `SyntaxError`: compilation_error=Syntax error in custom generated code or ModelNew not found
- `n8` `anchor_marker_drift`: anchor_marker_drift: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']
- `n9` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]
- `n10` `broken_anchor_markers`: broken_anchor_markers: expected=['helpers', 'cuda_cpp', 'cuda_cu', 'init_body', 'forward_stmt_1', 'forward_stmt_2']; observed=[]

## Code Artifacts

- run.json: `/data/dyj/KernelWeaver/runs/p1_single_cuda_cudallm_delib_main_20260513_124845/L1_P1_SquareMatmul_l1_p1/run.json`
- best_code.py: `/data/dyj/KernelWeaver/runs/p1_single_cuda_cudallm_delib_main_20260513_124845/L1_P1_SquareMatmul_l1_p1/best_code.py`
- best_node_id: `root`
- best plan: -
- best strategy: `None`
