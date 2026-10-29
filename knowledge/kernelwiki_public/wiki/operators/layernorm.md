---
id: "public-layernorm"
title: "Layer normalization domains"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["layernorm"]
tags: ["layernorm"]
priority: 80
sources: ["source-torch-norm"]
applicability: ["LayerNorm; inspect the full normalized_shape at runtime."]
advice: ["Normalize over all configured trailing dimensions and retain the biased variance, epsilon and optional elementwise affine parameters."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Inferred hypothesis: select a statistics strategy based on both domain size and available independent rows. A single last-axis assumption is insufficient. Numerical cancellation needs validation before replacing the variance algorithm.
