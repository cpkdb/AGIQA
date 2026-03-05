#!/usr/bin/env python3
"""
Render LLM generation instructions for positive-prompt backfill.

This module turns the all-dimensions backfill plan into concrete per-dimension
LLM inputs, without calling the API yet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "degradation_dimensions.md"
PLAN_PATH = DATA_DIR / "llm_backfill_plan_all_dimensions_v1.json"


def _load_plan() -> Dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _load_doc_dimensions() -> List[str]:
    text = DOC_PATH.read_text(encoding="utf-8")
    return [
        match.group(1)
        for match in re.finditer(r"^\| \*\*([a-z0-9_]+)\*\* \|", text, flags=re.MULTILINE)
    ]


def get_dimension_plan(dimension: str) -> Dict:
    """Return a single dimension plan entry from the all-dimensions config."""
    plan = _load_plan()
    for item in plan["dimensions"]:
        if item["dimension"] == dimension:
            return item
    raise KeyError(f"Unknown dimension: {dimension}")


def list_active_dimension_plans() -> List[Dict]:
    """Return only the active dimensions defined by degradation_dimensions.md, in doc order."""
    return [get_dimension_plan(dimension) for dimension in _load_doc_dimensions()]


def _resolve_coverage_mode(plan: Dict, dimension_plan: Dict) -> str:
    return dimension_plan.get(
        "coverage_mode",
        plan["default_coverage_mode_by_tier"][dimension_plan["tier"]],
    )


def _resolve_scene_mix(plan: Dict, dimension_plan: Dict) -> Dict[str, int]:
    if "scene_mix_override" in dimension_plan:
        return dict(dimension_plan["scene_mix_override"])

    total = dimension_plan["target_new_prompts"]
    template_family = dimension_plan["template_family"]
    default_mix = plan["default_scene_mix_policy_by_template_family"].get(template_family, {})
    if not default_mix:
        return {"objects_products_tools": total}

    items = list(default_mix.items())
    scene_mix: Dict[str, int] = {}
    assigned = 0
    for idx, (bucket, ratio) in enumerate(items):
        if idx == len(items) - 1:
            count = total - assigned
        else:
            count = int(round(total * ratio))
            assigned += count
        scene_mix[bucket] = count

    if sum(scene_mix.values()) != total:
        # Adjust the largest bucket to guarantee exact total.
        largest_bucket = max(scene_mix, key=scene_mix.get)
        scene_mix[largest_bucket] += total - sum(scene_mix.values())
    return scene_mix


def _render_system_prompt(plan: Dict, dimension_plan: Dict, scene_mix: Dict[str, int]) -> str:
    coverage_mode = _resolve_coverage_mode(plan, dimension_plan)
    template_family = dimension_plan["template_family"]
    complexity_mode = dimension_plan["complexity_mode"]
    macro_policy = dimension_plan.get("macro_taxonomy_policy_override")

    lines = [
        "You are generating high-quality positive prompts for a text-to-image model.",
        "Support the assigned downstream contrastive check without mentioning internal degradation labels.",
        f"Template family: {template_family}.",
        f"Coverage mode: {coverage_mode}.",
        f"Complexity mode: {complexity_mode}.",
        "Follow the configured macro taxonomy and micro-structure constraints.",
        "Do not use quality booster tokens such as masterpiece, best quality, 8k, or ultra detailed.",
        f"Keep the prompt length within {plan['length_range_words']['min']}-{plan['length_range_words']['max']} words.",
    ]

    if coverage_mode == "global_balanced":
        lines.append("Use global macro balance rather than a strict per-dimension scene quota.")
    else:
        lines.append("Respect the dimension-specific macro coverage policy.")

    if macro_policy:
        core = ", ".join(macro_policy.get("core", [])) or "none"
        optional = ", ".join(macro_policy.get("optional", [])) or "none"
        forbidden = ", ".join(macro_policy.get("forbidden", [])) or "none"
        lines.append(f"Use these core macro buckets: {core}.")
        lines.append(f"Use these optional macro buckets: {optional}.")
        lines.append(f"Avoid these forbidden macro buckets: {forbidden}.")

    lines.append("Scene mix targets:")
    for bucket, count in scene_mix.items():
        lines.append(f"- {bucket}: {count}")

    lines.append("Output only the generated prompts, without explanations or numbering unless explicitly requested.")
    return " ".join(lines)


def _render_job_system_prompt(
    plan: Dict,
    dimension_plan: Dict,
    macro_bucket: str,
    requested_prompts: int,
) -> str:
    coverage_mode = _resolve_coverage_mode(plan, dimension_plan)
    template_family = dimension_plan["template_family"]
    complexity_mode = dimension_plan["complexity_mode"]
    macro_policy = dimension_plan.get("macro_taxonomy_policy_override")

    lines = [
        "You are generating high-quality positive prompts for a text-to-image model.",
        "Support the assigned downstream contrastive check without mentioning internal degradation labels.",
        f"Template family: {template_family}.",
        f"Coverage mode: {coverage_mode}.",
        f"Complexity mode: {complexity_mode}.",
        f"This request is only for the macro bucket: {macro_bucket}.",
        "Follow the configured macro taxonomy and micro-structure constraints.",
        "Do not use quality booster tokens such as masterpiece, best quality, 8k, or ultra detailed.",
        f"Keep the prompt length within {plan['length_range_words']['min']}-{plan['length_range_words']['max']} words.",
    ]

    if coverage_mode == "global_balanced":
        lines.append("Use the current macro bucket only, while preserving globally balanced style diversity overall.")
    else:
        lines.append("Respect the dimension-specific macro coverage policy within this single macro bucket.")

    if macro_policy:
        core = ", ".join(macro_policy.get("core", [])) or "none"
        optional = ", ".join(macro_policy.get("optional", [])) or "none"
        forbidden = ", ".join(macro_policy.get("forbidden", [])) or "none"
        lines.append(f"Use these core macro buckets: {core}.")
        lines.append(f"Use these optional macro buckets: {optional}.")
        lines.append(f"Avoid these forbidden macro buckets: {forbidden}.")

    lines.append("This request target:")
    lines.append(f"- {macro_bucket}: {requested_prompts}")
    lines.append("Output only the generated prompts, without explanations or numbering unless explicitly requested.")
    return " ".join(lines)


def render_generation_prompt(dimension: str) -> Dict:
    """Render the per-dimension generation specification for a future LLM call."""
    plan = _load_plan()
    dimension_plan = get_dimension_plan(dimension)
    coverage_mode = _resolve_coverage_mode(plan, dimension_plan)
    scene_mix = _resolve_scene_mix(plan, dimension_plan)
    system_prompt = _render_system_prompt(plan, dimension_plan, scene_mix)

    return {
        "dimension": dimension,
        "coverage_mode": coverage_mode,
        "scene_mix": scene_mix,
        "system_prompt": system_prompt,
        "dimension_plan": dimension_plan,
    }


def render_job_prompt(dimension: str, macro_bucket: str, requested_prompts: int) -> Dict:
    """Render a per-job generation specification localized to one macro bucket."""
    plan = _load_plan()
    dimension_plan = get_dimension_plan(dimension)
    coverage_mode = _resolve_coverage_mode(plan, dimension_plan)
    scene_mix = {macro_bucket: requested_prompts}
    system_prompt = _render_job_system_prompt(
        plan=plan,
        dimension_plan=dimension_plan,
        macro_bucket=macro_bucket,
        requested_prompts=requested_prompts,
    )

    return {
        "dimension": dimension,
        "macro_bucket": macro_bucket,
        "coverage_mode": coverage_mode,
        "scene_mix": scene_mix,
        "system_prompt": system_prompt,
        "dimension_plan": dimension_plan,
    }


def render_batch_plan(dimensions: List[str]) -> List[Dict]:
    """Render multiple dimension generation specifications."""
    return [render_generation_prompt(dimension) for dimension in dimensions]


__all__ = [
    "get_dimension_plan",
    "list_active_dimension_plans",
    "render_batch_plan",
    "render_generation_prompt",
    "render_job_prompt",
]
