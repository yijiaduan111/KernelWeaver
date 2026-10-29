---
id: "public-maxpool"
title: "Maximum pooling boundaries"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["maxpool"]
tags: ["maxpool"]
priority: 80
sources: ["source-torch-pool"]
applicability: ["Regular max pooling; adaptive pooling and unpooling need separate contracts."]
advice: ["Respect window, stride, dilation, padding, ceil_mode and optional returned indices. Implicit padding uses negative infinity for floating-point maximum pooling."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Inferred hypothesis: compare output-oriented mappings and reuse only when windows overlap. Validate boundary outputs and index semantics. Derive every extent from actual parameters.
