---
id: "public-batchnorm"
title: "Batch normalization state and statistics"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["batchnorm"]
tags: ["batchnorm"]
priority: 80
sources: ["source-torch-batchnorm"]
applicability: ["Batch normalization only; check training, running buffers and affine mode."]
advice: ["Training statistics are computed per channel over batch and spatial dimensions. Normalization uses biased variance, while the running-variance update uses an unbiased estimate."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Evaluation may use running statistics; disabling their tracking changes this behavior. Preserve momentum, epsilon, affine parameters and state updates. Inferred hypothesis: fuse the elementwise transform after the required statistics are available, without silently dropping buffer mutations.
