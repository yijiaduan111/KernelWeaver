# KernelBench compatibility patch

The patch targets the exact submodule commit recorded in `.gitmodules` and Git.
Initialize the submodule, then apply it from the project root:

```bash
git submodule update --init --recursive
git -C KernelBench apply --check ../patches/kernelbench-compat.patch
git -C KernelBench apply ../patches/kernelbench-compat.patch
```

Apply it once per clean checkout. A second application fails instead of
silently stacking changes. To test whether it is already applied:

```bash
git -C KernelBench apply --reverse --check ../patches/kernelbench-compat.patch
```

The changes preserve integer and boolean input dtypes, apply the same input
handling to reference timing, make upstream API-generation clients optional
for local evaluation, and add an opt-in bounded-memory comparison.
`KERNELBENCH_CHECK_CHUNK_ELEMENTS` defaults to zero. When enabled, every element
is still checked with the original tolerance. It does not subsample outputs or
relax correctness. Difference-statistic accumulation may have different rounding
but does not determine pass/fail.

Report the upstream commit and patch application with experimental results.
The upstream license is retained in `licenses/KernelBench-MIT.txt`.
