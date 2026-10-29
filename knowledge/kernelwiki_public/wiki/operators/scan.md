---
id: "public-scan"
title: "Prefix scans and carry propagation"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["scan"]
tags: ["scan"]
priority: 85
sources: ["source-torch-ops","source-scan-algorithm","source-cuda-programming"]
applicability: ["Cumulative operators along a specified axis; respect additive versus multiplicative operation."]
advice: ["Cumsum is an inclusive sum along its requested dimension. Its output retains all prefix positions."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

For larger domains, scan chunks, scan chunk totals, then propagate carries. This is an algorithm family, not a chosen launch configuration. Maintain stream ordering between stages. Validate remainder chunks and floating-point reassociation; do not substitute a total reduction.
