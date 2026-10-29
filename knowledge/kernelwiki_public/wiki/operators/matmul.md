---
id: "public-matmul"
title: "Matrix multiplication and batching"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["matmul"]
tags: ["matmul"]
priority: 60
sources: ["source-torch-ops","source-cuda-practices"]
applicability: ["Matrix products, batched products and linear layers; derive rank and broadcasting from the current expression."]
advice: ["Preserve vector promotion, batch broadcasting, transpose and dtype semantics. Choose tiles only after reading strides and output dimensions."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Inferred optimization hypothesis: reuse operands through tiling when arithmetic intensity warrants it. Measure complete operator time and validate edges. Reduced-precision arithmetic requires explicit numerical permission.
