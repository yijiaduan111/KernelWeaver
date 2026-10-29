# Third-party notices

KernelWeaver's own code is distributed under the MIT license in `LICENSE`.

## KernelBench

The `KernelBench` submodule points to
[ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`.
It remains under its upstream MIT license, reproduced in
`licenses/KernelBench-MIT.txt`. The compatibility changes in
`patches/kernelbench-compat.patch` are identified separately and do not change
the benchmark's numerical tolerances or sample its output elements.

## Wiki sources

The public Wiki contains project-authored summaries with links to primary
documentation and papers. Source titles, source revisions and URLs are recorded
in each source page. Linked documentation, papers and associated implementations
retain their original rights and licenses. Linking to them does not relicense
their contents under this project's MIT license.

## Optional software and model weights

PyTorch, Triton, TileLang, CuTe/CUTLASS DSL, Transformers, API services and local
model weights are separate dependencies with their own licenses and terms.
No model weights, third-party agent repositories or API credentials are bundled.
