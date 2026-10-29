---
id: "public-convolution"
title: "Convolution and grouped connectivity"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["convolution"]
tags: ["convolution"]
priority: 75
sources: ["source-torch-conv","source-cuda-practices"]
applicability: ["Convolution with ordinary, grouped or depthwise channel connectivity."]
advice: ["Preserve cross-correlation, stride, dilation, padding mode, bias and group-specific weight layout. Depthwise convolution is a grouped special case, not an independent dense channel reduction."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Inferred hypothesis: select spatial/channel reuse according to groups and layout, then measure against the same reference. Guard tiles and derive output extents from parameters. Library-like tiling is not a correctness proof.
