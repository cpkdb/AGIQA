#!/usr/bin/env python3
"""
Build a human-readable summary for positive prompt LLM backfill allocation.

This is a reporting layer on top of the all-dimensions backfill plan. It resolves
default coverage modes and scene-mix allocations so the team can quickly inspect
which dimensions are being backfilled, how much, and into which macro buckets.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLAN_PATH = DATA_DIR / "llm_backfill_plan_all_dimensions_v1.json"
OUTPUT_PATH = DATA_DIR / "llm_backfill_summary_v1.json"
RENDERER_PATH = Path(__file__).resolve().parent / "positive_prompt_backfill_renderer.py"


def _load_plan() -> Dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _load_renderer():
    spec = spec_from_file_location("positive_prompt_backfill_renderer", RENDERER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _resolved_coverage_mode(plan: Dict, dimension_plan: Dict) -> str:
    return dimension_plan.get(
        "coverage_mode",
        plan["default_coverage_mode_by_tier"][dimension_plan["tier"]],
    )


def _build_dimension_summary(plan: Dict, renderer, dimension_plan: Dict) -> Dict:
    rendered = renderer.render_generation_prompt(dimension_plan["dimension"])
    macro_policy = dimension_plan.get(
        "macro_taxonomy_policy_override",
        {"core": [], "optional": [], "forbidden": []},
    )
    scene_mix = rendered["scene_mix"]

    return {
        "priority": dimension_plan["priority"],
        "tier": dimension_plan["tier"],
        "dimension": dimension_plan["dimension"],
        "current_rule_subpool_count": dimension_plan["current_rule_subpool_count"],
        "target_new_prompts": dimension_plan["target_new_prompts"],
        "template_family": dimension_plan["template_family"],
        "complexity_mode": dimension_plan["complexity_mode"],
        "coverage_mode": _resolved_coverage_mode(plan, dimension_plan),
        "llm_semantic_screening_after_merge": dimension_plan["llm_semantic_screening_after_merge"],
        "macro_policy": macro_policy,
        "scene_mix": scene_mix,
        "scene_mix_total": sum(scene_mix.values()),
    }


def build_summary() -> Dict:
    plan = _load_plan()
    renderer = _load_renderer()

    tier_totals: Dict[str, int] = {}
    coverage_mode_totals: Dict[str, int] = {}
    dimensions: Dict[str, Dict] = {}

    active_dimensions = renderer.list_active_dimension_plans()

    for item in active_dimensions:
        dimension_summary = _build_dimension_summary(plan, renderer, item)
        dimensions[item["dimension"]] = dimension_summary

        tier = dimension_summary["tier"]
        tier_totals[tier] = tier_totals.get(tier, 0) + dimension_summary["target_new_prompts"]

        coverage_mode = dimension_summary["coverage_mode"]
        coverage_mode_totals[coverage_mode] = (
            coverage_mode_totals.get(coverage_mode, 0) + dimension_summary["target_new_prompts"]
        )

    return {
        "schema_version": plan["schema_version"],
        "source_plan": PLAN_PATH.name,
        "dimension_count": len(active_dimensions),
        "total_target_new_prompts": sum(
            item["target_new_prompts"] for item in active_dimensions
        ),
        "tier_totals": tier_totals,
        "coverage_mode_totals": coverage_mode_totals,
        "dimensions": dimensions,
    }


def build_priority_queue(limit: int | None = None) -> List[Dict]:
    summary = build_summary()
    queue = sorted(
        summary["dimensions"].values(),
        key=lambda item: (item["priority"], -item["target_new_prompts"], item["dimension"]),
    )
    if limit is not None:
        return queue[:limit]
    return queue


def write_summary(output_path: Path = OUTPUT_PATH) -> Path:
    summary = build_summary()
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    path = write_summary()
    print(path)
