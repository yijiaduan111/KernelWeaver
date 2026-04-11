from __future__ import annotations

from textwrap import dedent

from .models import StrategySpec, TaskSpec, TestCase


def build_triton_tasks() -> list[TaskSpec]:
    torch = _require_torch()
    _require_cuda(torch)
    square_task = TaskSpec(
        name="elementwise_square",
        description="Triton microbenchmark that squares a CUDA tensor elementwise.",
        source_code=dedent(
            """
            import torch
            import triton
            import triton.language as tl

            def solve(x):
                # <<<IMPROVE:body>>>
                return x * x
                # <<<END_IMPROVE>>>
            """
        ).strip()
        + "\n",
        reference_code=dedent(
            """
            import torch

            def reference_solve(x):
                return x * x
            """
        ).strip()
        + "\n",
        function_name="solve",
        reference_function_name="reference_solve",
        test_cases=[
            TestCase(label="small", args=[torch.linspace(-4, 4, 1024, device="cuda", dtype=torch.float32)]),
            TestCase(label="medium", args=[torch.randn(4096, device="cuda", dtype=torch.float32)]),
        ],
        benchmark_cases=[
            TestCase(label="bench-a", args=[torch.randn(1 << 20, device="cuda", dtype=torch.float32)]),
            TestCase(label="bench-b", args=[torch.randn((1 << 20) + 257, device="cuda", dtype=torch.float32)]),
        ],
        tags=["triton", "gpu", "single-op"],
        strategy_catalog=[
            StrategySpec(
                name="triton_elementwise_kernel",
                anchor_name="body",
                strategy_summary="Use a Triton kernel to square the tensor in parallel on GPU.",
                instruction="Launch a Triton kernel that loads, squares, and stores the tensor with bounds masking.",
                expected_gain="Move the elementwise work into a custom Triton kernel.",
                good_body=dedent(
                    """
                    output = torch.empty_like(x)
                    n_elements = x.numel()

                    @triton.jit
                    def square_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                        pid = tl.program_id(0)
                        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_elements
                        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                        tl.store(out_ptr + offsets, values * values, mask=mask)

                    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                    square_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
                    return output
                    """
                ),
                broken_body=dedent(
                    """
                    output = torch.empty_like(x)
                    n_elements = x.numel()

                    @triton.jit
                    def square_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                        pid = tl.program_id(0)
                        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_elements
                        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                        tl.store(out_ptr + offsets, values + values, mask=mask)

                    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                    square_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
                    return output
                    """
                ),
                debug_body=dedent(
                    """
                    output = torch.empty_like(x)
                    n_elements = x.numel()

                    @triton.jit
                    def square_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                        pid = tl.program_id(0)
                        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_elements
                        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                        tl.store(out_ptr + offsets, values * values, mask=mask)

                    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                    square_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
                    return output
                    """
                ),
                broken_failure_type="correctness_error",
            )
        ],
    )

    fused_task = TaskSpec(
        name="triton_fused_affine_relu",
        description="Triton microbenchmark that fuses bias, scale, and ReLU into one GPU kernel.",
        source_code=dedent(
            """
            import torch
            import triton
            import triton.language as tl

            def solve(x, bias, scale):
                # <<<IMPROVE:body>>>
                return torch.relu((x + bias) * scale)
                # <<<END_IMPROVE>>>
            """
        ).strip()
        + "\n",
        reference_code=dedent(
            """
            import torch

            def reference_solve(x, bias, scale):
                return torch.relu((x + bias) * scale)
            """
        ).strip()
        + "\n",
        function_name="solve",
        reference_function_name="reference_solve",
        test_cases=[
            TestCase(label="small", args=[torch.randn(2048, device="cuda", dtype=torch.float32), 0.25, 1.5]),
            TestCase(label="medium", args=[torch.randn(8192, device="cuda", dtype=torch.float32), -0.5, 0.75]),
        ],
        benchmark_cases=[
            TestCase(label="bench-a", args=[torch.randn(1 << 20, device="cuda", dtype=torch.float32), 0.125, 1.1]),
            TestCase(label="bench-b", args=[torch.randn((1 << 20) + 511, device="cuda", dtype=torch.float32), -0.75, 0.9]),
        ],
        tags=["triton", "gpu", "fused-op"],
        strategy_catalog=[
            StrategySpec(
                name="triton_fused_affine_relu_kernel",
                anchor_name="body",
                strategy_summary="Fuse bias, scale, and ReLU into a single Triton kernel.",
                instruction="Implement one Triton kernel that loads x, applies affine transform, then ReLU, and stores the result.",
                expected_gain="Eliminate extra eager passes and run the fused math on GPU.",
                good_body=dedent(
                    """
                    output = torch.empty_like(x)
                    n_elements = x.numel()

                    @triton.jit
                    def fused_affine_relu_kernel(x_ptr, out_ptr, bias, scale, n_elements, BLOCK_SIZE: tl.constexpr):
                        pid = tl.program_id(0)
                        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_elements
                        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                        fused = (values + bias) * scale
                        relu = tl.maximum(fused, 0.0)
                        tl.store(out_ptr + offsets, relu, mask=mask)

                    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                    fused_affine_relu_kernel[grid](x, output, bias, scale, n_elements, BLOCK_SIZE=1024)
                    return output
                    """
                ),
                broken_body=dedent(
                    """
                    output = torch.empty_like(x)
                    n_elements = x.numel()

                    @triton.jit
                    def fused_affine_relu_kernel(x_ptr, out_ptr, bias, scale, n_elements, BLOCK_SIZE: tl.constexpr):
                        pid = tl.program_id(0)
                        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_elements
                        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                        fused = values + bias * scale
                        relu = tl.maximum(fused, 0.0)
                        tl.store(out_ptr + offsets, relu, mask=mask)

                    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                    fused_affine_relu_kernel[grid](x, output, bias, scale, n_elements, BLOCK_SIZE=1024)
                    return output
                    """
                ),
                debug_body=dedent(
                    """
                    output = torch.empty_like(x)
                    n_elements = x.numel()

                    @triton.jit
                    def fused_affine_relu_kernel(x_ptr, out_ptr, bias, scale, n_elements, BLOCK_SIZE: tl.constexpr):
                        pid = tl.program_id(0)
                        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_elements
                        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                        fused = (values + bias) * scale
                        relu = tl.maximum(fused, 0.0)
                        tl.store(out_ptr + offsets, relu, mask=mask)

                    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                    fused_affine_relu_kernel[grid](x, output, bias, scale, n_elements, BLOCK_SIZE=1024)
                    return output
                    """
                ),
                broken_failure_type="correctness_error",
            )
        ],
    )
    reduction_task = TaskSpec(
        name="triton_rowwise_sum",
        description="Triton microbenchmark that reduces each matrix row into a single sum value.",
        source_code=dedent(
            """
            import torch
            import triton
            import triton.language as tl

            def solve(x):
                # <<<IMPROVE:body>>>
                return x.sum(dim=1)
                # <<<END_IMPROVE>>>
            """
        ).strip()
        + "\n",
        reference_code=dedent(
            """
            import torch

            def reference_solve(x):
                return x.sum(dim=1)
            """
        ).strip()
        + "\n",
        function_name="solve",
        reference_function_name="reference_solve",
        test_cases=[
            TestCase(label="small", args=[torch.randn((128, 256), device="cuda", dtype=torch.float32)]),
            TestCase(label="medium", args=[torch.randn((256, 512), device="cuda", dtype=torch.float32)]),
        ],
        benchmark_cases=[
            TestCase(label="bench-a", args=[torch.randn((1024, 512), device="cuda", dtype=torch.float32)]),
            TestCase(label="bench-b", args=[torch.randn((2048, 512), device="cuda", dtype=torch.float32)]),
        ],
        tags=["triton", "gpu", "reduction"],
        strategy_catalog=[
            StrategySpec(
                name="triton_rowwise_sum_kernel",
                anchor_name="body",
                strategy_summary="Reduce each row with a Triton kernel that accumulates one program per row.",
                instruction="Launch a Triton kernel over rows, load one row block, sum across the columns, and store a scalar per row.",
                expected_gain="Move row reduction work into a custom Triton kernel while keeping one output per row.",
                good_body=dedent(
                    """
                    output = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
                    n_rows, n_cols = x.shape

                    @triton.jit
                    def rowwise_sum_kernel(x_ptr, out_ptr, stride_row, stride_col, n_cols, BLOCK_SIZE: tl.constexpr):
                        row_idx = tl.program_id(0)
                        offsets = tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_cols
                        row_ptr = x_ptr + row_idx * stride_row + offsets * stride_col
                        values = tl.load(row_ptr, mask=mask, other=0.0)
                        tl.store(out_ptr + row_idx, tl.sum(values, axis=0))

                    grid = (n_rows,)
                    rowwise_sum_kernel[grid](x, output, x.stride(0), x.stride(1), n_cols, BLOCK_SIZE=512)
                    return output
                    """
                ),
                broken_body=dedent(
                    """
                    output = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
                    n_rows, n_cols = x.shape

                    @triton.jit
                    def rowwise_sum_kernel(x_ptr, out_ptr, stride_row, stride_col, n_cols, BLOCK_SIZE: tl.constexpr):
                        row_idx = tl.program_id(0)
                        offsets = tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_cols
                        row_ptr = x_ptr + row_idx * stride_row + offsets * stride_col
                        values = tl.load(row_ptr, mask=mask, other=0.0)
                        tl.store(out_ptr + row_idx, tl.max(values, axis=0))

                    grid = (n_rows,)
                    rowwise_sum_kernel[grid](x, output, x.stride(0), x.stride(1), n_cols, BLOCK_SIZE=512)
                    return output
                    """
                ),
                debug_body=dedent(
                    """
                    output = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
                    n_rows, n_cols = x.shape

                    @triton.jit
                    def rowwise_sum_kernel(x_ptr, out_ptr, stride_row, stride_col, n_cols, BLOCK_SIZE: tl.constexpr):
                        row_idx = tl.program_id(0)
                        offsets = tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_cols
                        row_ptr = x_ptr + row_idx * stride_row + offsets * stride_col
                        values = tl.load(row_ptr, mask=mask, other=0.0)
                        tl.store(out_ptr + row_idx, tl.sum(values, axis=0))

                    grid = (n_rows,)
                    rowwise_sum_kernel[grid](x, output, x.stride(0), x.stride(1), n_cols, BLOCK_SIZE=512)
                    return output
                    """
                ),
                broken_failure_type="correctness_error",
            )
        ],
    )
    return [square_task, fused_task, reduction_task]



def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"torch is required to build Triton tasks: {exc}") from exc
    return torch



def _require_cuda(torch) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build Triton tasks.")
