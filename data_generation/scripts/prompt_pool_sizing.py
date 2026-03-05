#!/usr/bin/env python3
"""
Prompt pool sizing utilities.

This module estimates how many positive prompts are needed before building a
dimension-aware prompt pool, and ranks candidate prompt sources by coverage fit.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set


def _normalize_weights(dimension_weights: Dict[str, float]) -> Dict[str, float]:
    if not dimension_weights:
        return {}

    total_weight = sum(max(weight, 0.0) for weight in dimension_weights.values())
    if total_weight <= 0:
        equal_weight = 1.0 / len(dimension_weights)
        return {dimension: equal_weight for dimension in dimension_weights}

    return {
        dimension: max(weight, 0.0) / total_weight
        for dimension, weight in dimension_weights.items()
    }


def estimate_prompt_pool_size(
    total_pairs: int,
    num_negatives_per_positive: int,
    dimension_weights: Dict[str, float],
    constrained_dimensions: Optional[Iterable[str]] = None,
    constrained_multiplier: float = 1.5,
    min_per_dimension: int = 10,
) -> Dict:
    if total_pairs <= 0:
        raise ValueError("total_pairs must be positive")
    if num_negatives_per_positive <= 0:
        raise ValueError("num_negatives_per_positive must be positive")

    normalized_weights = _normalize_weights(dimension_weights)
    constrained_set: Set[str] = set(constrained_dimensions or [])

    target_positive_pool_size = math.ceil(total_pairs / num_negatives_per_positive)
    per_dimension = {}

    for dimension, weight in normalized_weights.items():
        estimated_pairs = int(round(total_pairs * weight))
        base_required = math.ceil(estimated_pairs / num_negatives_per_positive)
        is_constrained = dimension in constrained_set

        if is_constrained:
            required_positive_prompts = math.ceil(base_required * constrained_multiplier)
        else:
            required_positive_prompts = base_required

        required_positive_prompts = max(required_positive_prompts, min_per_dimension)

        per_dimension[dimension] = {
            "weight": weight,
            "estimated_pairs": estimated_pairs,
            "required_positive_prompts": required_positive_prompts,
            "is_constrained": is_constrained,
        }

    return {
        "total_pairs": total_pairs,
        "num_negatives_per_positive": num_negatives_per_positive,
        "target_positive_pool_size": target_positive_pool_size,
        "per_dimension": per_dimension,
    }


def rank_prompt_sources(
    sources: Sequence[Dict],
    target_dimensions: Sequence[str],
) -> List[Dict]:
    ranked = []
    target_dimensions = list(target_dimensions)

    for source in sources:
        coverage = source.get("dimension_coverage", {})
        prompt_count = int(source.get("prompt_count", 0))

        if target_dimensions:
            avg_coverage = sum(
                float(coverage.get(dimension, 0.0)) for dimension in target_dimensions
            ) / len(target_dimensions)
        else:
            avg_coverage = 0.0

        selection_score = avg_coverage * math.sqrt(max(prompt_count, 0))

        enriched = dict(source)
        enriched["avg_target_coverage"] = avg_coverage
        enriched["selection_score"] = selection_score
        ranked.append(enriched)

    ranked.sort(key=lambda item: item["selection_score"], reverse=True)
    return ranked


__all__ = [
    "estimate_prompt_pool_size",
    "rank_prompt_sources",
]
