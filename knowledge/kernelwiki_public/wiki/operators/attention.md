---
id: "public-attention"
title: "Attention compositions and IO-aware tiling"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["attention"]
tags: ["attention"]
priority: 100
sources: ["source-torch-sdpa","source-flashattention"]
applicability: ["SDPA, or a possible matrix-product/softmax composition. Co-occurring API calls are only a retrieval hint; confirm the dataflow."]
advice: ["Verify scale, mask convention, softmax axis, head broadcasting, dropout and output shape. Boolean SDPA masks select participating elements; additive masks alter logits."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

FlashAttention motivates tiling to reduce intermediate memory traffic. A composed matmul plus softmax is not automatically equivalent to SDPA. Confirm semantics before fusion, and validate partial tiles and numerical behavior. No architecture-specific kernel or speed claim is supplied.
