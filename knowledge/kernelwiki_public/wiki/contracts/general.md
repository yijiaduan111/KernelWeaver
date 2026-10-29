---
id: "public-contract"
title: "Contract checklist for an uncovered operator"
type: "contract"
knowledge_scope: "public_general"
origin: "public_documentation"
status: "reviewed"
confidence: "inferred"
reproducibility: "documentation"
architectures: ["sm86"]
backends: ["cuda"]
tags: ["contract"]
sources: ["source-cuda-programming","source-cuda-practices"]
advice: ["No specialized operator page matched. Derive semantics from the supplied task and request additional primary documentation before inventing an operator algorithm."]
---

Check input and output dtypes, strides, broadcasting, state mutation, indexing width, aliasing and synchronization. Validate a minimal candidate before performance work. Measure the full operation with consistent warmup and timing.

This page supplies a review procedure, not specialized coverage. An uncovered operator must remain visible in coverage reports.
