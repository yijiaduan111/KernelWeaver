---
id: "public-groupnorm"
title: "Group normalization domains"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["groupnorm"]
tags: ["groupnorm"]
priority: 80
sources: ["source-torch-norm"]
applicability: ["GroupNorm; channels must satisfy the configured grouping."]
advice: ["Compute statistics independently for each sample and group, including the spatial domain. Preserve biased variance, epsilon and channel-wise affine behavior."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Inferred hypothesis: map work to actual group domains. Reusing batch-normalization reductions would mix independent samples and change the operator.
