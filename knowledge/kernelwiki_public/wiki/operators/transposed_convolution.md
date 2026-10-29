---
id: "public-transposed_convolution"
title: "Transposed convolution indexing"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["transposed_convolution"]
tags: ["transposed_convolution"]
priority: 85
sources: ["source-torch-conv"]
applicability: ["Transposed convolution in the declared spatial dimensionality."]
advice: ["Preserve transposed weight layout, groups, stride, dilation, padding and output_padding. The last parameter resolves output extent and does not append a zero-filled output border."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Inferred implementation choices: gather all valid input contributors per output, or safely accumulate scatter contributions. Derive inverse coordinates and divisibility conditions; never assume one input contributes to a disjoint output.
