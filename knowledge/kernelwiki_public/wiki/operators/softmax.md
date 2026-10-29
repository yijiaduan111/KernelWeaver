---
id: "public-softmax"
title: "Softmax axis and stable normalization"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["softmax"]
tags: ["softmax"]
priority: 65
sources: ["source-torch-activation"]
applicability: ["Softmax or LogSoftmax along the requested axis."]
advice: ["Preserve axis, output dtype and the distinction between probabilities and log probabilities."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Inferred numerical construction: subtract the maximum before exponentiation and normalize with a sum on that same domain. Handle nonfinite values according to the task contract. Tile sizes are workload decisions, and changing them needs fresh correctness validation.
