from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..deliberation.schema import DeliberationStrategy, StrategyPortfolio
from ..feedback.schema import FeedbackState
from .catalog import MethodCatalogEntry, load_method_catalog
from .schema import MemoryMethodCard, MemoryProfile


def build_memory_profile(
    task: Any,
    *,
    enabled: bool = True,
    mode: str = "expert_memory_v0",
    max_primary_cards: int = 4,
    max_challenger_cards: int = 3,
) -> MemoryProfile | None:
    op_type = _task_op_type(task)
    backend = str(getattr(task, "backend", "") or "unknown")
    if not enabled:
        return MemoryProfile(
            enabled=False,
            mode=mode,
            backend=backend,
            op_type=op_type,
            source="machine_check",
            notes=["Long-term memory is disabled by configuration."],
        )
    if backend != "cuda":
        return MemoryProfile(
            enabled=False,
            mode=mode,
            backend=backend,
            op_type=op_type,
            source="machine_check",
            notes=["Expert long-term memory is currently scoped to CUDA tasks only."],
        )

    diagnostics = getattr(task, "diagnostics_profile", None)
    machine_check = getattr(diagnostics, "machine_check_profile", None)
    if diagnostics is None or machine_check is None or not machine_check.enabled:
        notes = list(getattr(diagnostics, "notes", []) or [])
        if not notes:
            notes = ["Machine-check diagnostics are unavailable; memory bootstrap is disabled instead of guessing methods."]
        return MemoryProfile(
            enabled=False,
            mode=mode,
            backend=backend,
            op_type=op_type,
            source="machine_check",
            notes=notes,
        )

    catalog = load_method_catalog()
    anchors = _recommended_anchors(task)
    allowed_methods = [method_id for method_id in machine_check.allowed_methods if method_id in catalog]
    forbidden_methods = [method_id for method_id in machine_check.forbidden_methods if method_id]
    primary_ids = allowed_methods[:max_primary_cards]
    challenger_ids = [method_id for method_id in allowed_methods[max_primary_cards:] if method_id not in primary_ids][:max_challenger_cards]
    bootstrap_cards = [
        _build_card(catalog, method_id, anchors, priority=max(1, 6 - index))
        for index, method_id in enumerate(primary_ids, start=1)
    ]
    challenger_cards = [
        _build_card(catalog, method_id, anchors, priority=max(1, 4 - index))
        for index, method_id in enumerate(challenger_ids, start=1)
        if method_id not in {card.method_id for card in bootstrap_cards}
    ]
    preferred_methods = [card.method_id for card in bootstrap_cards[:2]]
    notes = list(machine_check.notes)
    if machine_check.bottleneck_id:
        notes.append(f"machine_check bottleneck={machine_check.bottleneck_id} case={machine_check.case_id or 'unknown'}.")
    feedback_digest = {
        "phase": "bootstrap",
        "champion_method": None,
        "plateau_detected": False,
        "recent_positive_methods": [],
        "recent_negative_methods": [],
        "recent_positive_focuses": [],
        "recent_negative_focuses": [],
        "tier": machine_check.tier,
        "bottleneck_id": machine_check.bottleneck_id,
        "case_id": machine_check.case_id,
    }
    return MemoryProfile(
        enabled=bool(bootstrap_cards or challenger_cards),
        mode=mode,
        backend=backend,
        op_type=op_type,
        source="machine_check",
        bottleneck_id=machine_check.bottleneck_id,
        case_id=machine_check.case_id,
        allowed_methods=list(allowed_methods),
        forbidden_methods=list(forbidden_methods),
        bootstrap_cards=bootstrap_cards,
        challenger_cards=challenger_cards,
        preferred_methods=preferred_methods,
        blocked_methods=list(forbidden_methods),
        feedback_digest=feedback_digest,
        notes=_ordered_unique(notes),
    )


def refresh_memory_profile(
    task: Any,
    feedback_state: FeedbackState | None,
    *,
    top_k: int = 3,
) -> MemoryProfile | None:
    profile = getattr(task, "memory_profile", None)
    if profile is None or not profile.enabled or feedback_state is None:
        return profile
    champion_method = str(feedback_state.champion.mutation_family or "").strip() or None
    positive_methods = _ordered_unique(
        [str(item) for item in feedback_state.recent_successful_mutation_families if str(item).strip()]
        + [str(item.get("mutation_family")) for item in feedback_state.champion.recent_positive_mutations if str(item.get("mutation_family") or "").strip()]
    )[:top_k]
    negative_methods = _ordered_unique(
        [str(item) for item in feedback_state.recent_failed_mutation_families if str(item).strip()]
        + [str(item.get("mutation_family")) for item in feedback_state.champion.recent_negative_mutations if str(item.get("mutation_family") or "").strip()]
    )[:top_k]
    preferred_methods = _ordered_unique(([champion_method] if champion_method else []) + positive_methods + list(profile.preferred_methods))
    preferred_methods = [item for item in preferred_methods if not profile.allowed_methods or item in profile.allowed_methods][: max(1, top_k + 1)]
    blocked_methods = _ordered_unique(list(profile.forbidden_methods) + [item for item in negative_methods if item not in preferred_methods])
    notes = list(profile.notes)
    if feedback_state.plateau_detected:
        notes.append("Champion lineage plateaued; promote challenger families for the next plan selection.")
    if champion_method:
        notes.append(f"Current champion family: {champion_method}.")
    updated = replace(
        profile,
        preferred_methods=preferred_methods,
        blocked_methods=blocked_methods,
        feedback_digest={
            "phase": "feedback_refresh",
            "champion_method": champion_method,
            "plateau_detected": bool(feedback_state.plateau_detected),
            "recent_positive_methods": positive_methods,
            "recent_negative_methods": negative_methods,
            "recent_positive_focuses": _recent_focuses(feedback_state.champion.recent_positive_mutations, top_k),
            "recent_negative_focuses": _recent_focuses(feedback_state.champion.recent_negative_mutations, top_k),
            "compile_rate": round(float(feedback_state.compile_rate), 3),
            "correct_rate": round(float(feedback_state.correct_rate), 3),
            "best_speedup": round(float(feedback_state.best_speedup), 3) if feedback_state.best_speedup is not None else None,
            "tier": profile.feedback_digest.get("tier"),
            "bottleneck_id": profile.feedback_digest.get("bottleneck_id"),
            "case_id": profile.feedback_digest.get("case_id"),
        },
        notes=_ordered_unique(notes)[-8:],
    )
    setattr(task, "memory_profile", updated)
    portfolio = getattr(task, "strategy_portfolio", None)
    if portfolio is not None:
        setattr(task, "strategy_portfolio", rebalance_strategy_portfolio(portfolio, updated, feedback_state))
    return updated


def rebalance_strategy_portfolio(
    portfolio: StrategyPortfolio,
    profile: MemoryProfile,
    feedback_state: FeedbackState | None,
) -> StrategyPortfolio:
    if not portfolio.strategies:
        return portfolio
    attempted = {
        str(item.strategy_name)
        for item in getattr(feedback_state, "recent_attempts", []) or []
        if getattr(item, "strategy_name", None)
    }
    preferred = set(profile.preferred_methods)
    blocked = set(profile.blocked_methods)
    challengers = {card.method_id for card in profile.challenger_cards}
    allowed = set(profile.allowed_methods)
    champion_method = str((feedback_state.champion.mutation_family if feedback_state is not None else "") or "").strip() or None
    scored: list[tuple[float, int, DeliberationStrategy]] = []
    for index, strategy in enumerate(portfolio.strategies):
        base_priority = float(strategy.priority)
        avg_review = sum(strategy.model_scores.values()) / len(strategy.model_scores) if strategy.model_scores else 0.0
        methods = list(strategy.memory_methods)
        family = methods[0] if methods else None
        score = base_priority + avg_review
        if family and allowed and family not in allowed:
            score -= 5.0
        if family in preferred:
            score += 3.0
        if family in blocked:
            score -= 3.0
        if feedback_state is not None and feedback_state.plateau_detected:
            if family in challengers and family != champion_method:
                score += 2.0
            if champion_method and family == champion_method:
                score -= 0.5
        if strategy.strategy_id not in attempted:
            score += 0.25
        strategy.priority = max(1, min(9, int(round(score))))
        scored.append((score, -index, strategy))
    portfolio.strategies = [item for _score, _index, item in sorted(scored, key=lambda row: (row[0], row[1]), reverse=True)]
    return portfolio


def infer_memory_methods(summary: str, profile: MemoryProfile | None) -> list[str]:
    if profile is None:
        return []
    text = str(summary or "").lower()
    rules = [
        ("vector", "Vectorization_Refinement"),
        ("coalesc", "Improve_Coalescing_and_TransactionSize"),
        ("alignment", "Alignment_and_Tail_Minimization"),
        ("tail", "Alignment_and_Tail_Minimization"),
        ("launch", "Launch_Tuning"),
        ("occupancy", "Increase_Occupancy_if_limited"),
        ("til", "SharedMemoryTiling"),
        ("shared", "SharedMemoryTiling"),
        ("register", "RegisterBlocking"),
        ("fuse", "KernelFusion"),
        ("fusion", "KernelFusion"),
        ("tensor core", "TensorCore_or_CUBLASLT"),
        ("cublas", "TensorCore_or_CUBLASLT"),
        ("instruction", "Reduce_Instruction_Count"),
        ("math", "Vectorized_Math"),
        ("transc", "Approximate_Transcendentals_or_LUT"),
    ]
    matched = [method_id for token, method_id in rules if token in text]
    if matched:
        filtered = [method_id for method_id in _ordered_unique(matched) if not profile.allowed_methods or method_id in profile.allowed_methods]
        return filtered[:2]
    primary = [card.method_id for card in profile.bootstrap_cards]
    return primary[:1]


def card_map(profile: MemoryProfile | None) -> dict[str, MemoryMethodCard]:
    if profile is None:
        return {}
    return {
        card.method_id: card
        for card in [*profile.bootstrap_cards, *profile.challenger_cards]
    }


def _task_op_type(task: Any) -> str:
    semantic_profile = getattr(task, "semantic_profile", None)
    return str(getattr(semantic_profile, "op_type", "unknown") or "unknown")


def _recommended_anchors(task: Any) -> list[str]:
    semantic_profile = getattr(task, "semantic_profile", None)
    anchors = list(getattr(semantic_profile, "recommended_anchors", []) or [])
    grounded = [str(getattr(region, "anchor_name", "")) for region in getattr(task, "grounded_regions", [])]
    output: list[str] = []
    for item in [*anchors, *grounded]:
        text = str(item).strip()
        if text and text not in output:
            output.append(text)
    return output[:6]


def _build_card(
    catalog: dict[str, MethodCatalogEntry],
    method_id: str,
    anchors: list[str],
    *,
    priority: int,
) -> MemoryMethodCard:
    entry = catalog[method_id]
    return MemoryMethodCard(
        method_id=entry.method_id,
        title=entry.title,
        summary=entry.summary,
        source=entry.source,
        target_anchors=_card_target_anchors(method_id, anchors),
        why_now=[],
        implementation_hints=list(entry.mechanism_requirements[:6]),
        forbidden_patterns=list(entry.forbidden_patterns[:4]),
        expected_metric_change=list(entry.expected_metric_change[:4]),
        priority=priority,
    )


def _card_target_anchors(method_id: str, anchors: list[str]) -> list[str]:
    output: list[str] = []
    preferred_backend = {"cuda_cu", "cuda_cpp", "user_helpers"}
    if method_id in {"KernelFusion", "CUDA_Graph_Capture_Replay_StaticBuffers", "MultiKernelScheduling"}:
        for anchor in anchors:
            if anchor.startswith("forward_stmt_") or anchor == "user_helpers":
                output.append(anchor)
    for anchor in anchors:
        if anchor in preferred_backend and anchor not in output:
            output.append(anchor)
    for anchor in anchors:
        if anchor not in output:
            output.append(anchor)
    return output[:4]


def _recent_focuses(records: list[dict[str, Any]], top_k: int) -> list[str]:
    return _ordered_unique([str(item.get("single_change_focus")) for item in records if str(item.get("single_change_focus") or "").strip()])[:top_k]


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
