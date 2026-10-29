---
id: "public-cross_entropy"
title: "Cross entropy target and reduction modes"
type: "kernel"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "source-reported"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
operator_families: ["cross_entropy"]
tags: ["cross_entropy"]
priority: 90
sources: ["source-torch-loss"]
applicability: ["CrossEntropyLoss and equivalent functional calls; inspect target representation."]
advice: ["Class-index and probability targets have distinct contracts. Preserve weights, label smoothing, ignore_index where applicable, and none/sum/mean reduction. For weighted class-index targets, mean divides by the applicable target-weight total."]
acceptance: ["Check the complete operator contract before timing; retain a change only after measured validation."]
---

Inferred hypothesis: combine stable log-probability computation with loss evaluation when semantics permit. Retain integer indices. Check all-ignored and boundary-label cases against the reference instead of silently choosing a new convention.
