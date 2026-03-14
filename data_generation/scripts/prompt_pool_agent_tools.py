#!/usr/bin/env python3
"""
Deterministic tools for the Prompt Pool Agent.

The first version is analysis-only:
- resolve current active prompt pools and dimension subpool indices
- scan prompt-pool artifacts under inventory roots
- classify active / fallback / stale pool artifacts
- build a 32-dimension routing view from a static routing config
- produce screening-plan and cleanup-candidate artifacts
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import taxonomy_allocation_tools as taxonomy_tools


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
DEFAULT_ROUTING_CONFIG_PATH = CONFIG_DIR / "prompt_pool_routing_v1.json"
DEFAULT_SCREENING_SPEC_PATH = CONFIG_DIR / "prompt_pool_screening_spec_v1.json"
DEFAULT_BUILD_TARGETS_PATH = CONFIG_DIR / "prompt_pool_build_targets_v1.json"
DEFAULT_TAXONOMY_PATH = taxonomy_tools.DEFAULT_TAXONOMY_PATH
DEFAULT_RUNTIME_RESOURCE_CANDIDATES = taxonomy_tools.DEFAULT_RUNTIME_RESOURCE_CANDIDATES
DEFAULT_MODEL_IDS = taxonomy_tools.DEFAULT_MODEL_IDS

DEFAULT_POOL_INVENTORY_ROOTS = [
    Path("/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full"),
    Path("/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs"),
]

DEFAULT_STALE_CANDIDATE_NAMES = [
    "sd35_turbo_clipsafe_v2_tmp",
    "targeted_dimension_subpools_cleaned_v1",
    "sd35_turbo_targeted_dimension_subpools_clipsafe_v1",
]

POOL_NAME_HINTS = (
    "pool",
    "subpool",
    "screened",
    "clipsafe",
    "working_pool",
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def resolve_active_prompt_pools(
    *,
    resource_candidates: taxonomy_tools.RuntimeResourceCandidates = DEFAULT_RUNTIME_RESOURCE_CANDIDATES,
    model_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    return taxonomy_tools.resolve_runtime_resources(
        resource_candidates=resource_candidates,
        model_ids=model_ids,
    )


def inspect_active_pool_coverage(
    *,
    resolved_resources: Dict[str, Dict[str, Optional[str]]],
    dimensions: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    coverage_by_model: Dict[str, Dict[str, Any]] = {}
    for model_id, resource_info in resolved_resources.items():
        source_prompts = resource_info.get("source_prompts")
        dimension_subpool_index = resource_info.get("dimension_subpool_index")
        if not source_prompts or not dimension_subpool_index:
            coverage_by_model[model_id] = {
                "source_prompts": source_prompts,
                "dimension_subpool_index": dimension_subpool_index,
                "total_source_prompts": 0,
                "dimension_counts": {},
                "warnings": {
                    "low_pool_dims": [],
                    "missing_dims": list(dimensions),
                },
            }
            continue
        coverage_by_model[model_id] = taxonomy_tools.inspect_pool_coverage(
            source_prompts_path=Path(source_prompts),
            dimension_subpool_index=Path(dimension_subpool_index),
            dimensions=dimensions,
        )
    return coverage_by_model


def _looks_like_pool_artifact(path: Path) -> bool:
    name = path.name.lower()
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return True
    return any(hint in name for hint in POOL_NAME_HINTS)


def _build_inventory_entry(path: Path) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "type": "directory" if path.is_dir() else "file",
        "contains_index": False,
        "prompt_count": None,
    }
    if path.is_dir():
        entry["contains_index"] = (path / "index.json").exists()
    elif path.suffix.lower() == ".jsonl":
        entry["prompt_count"] = _count_jsonl_rows(path)
    return entry


def scan_prompt_pool_inventory(
    *,
    inventory_roots: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    roots = [Path(root) for root in (inventory_roots or DEFAULT_POOL_INVENTORY_ROOTS)]
    entries: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()

    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if not _looks_like_pool_artifact(child):
                continue
            key = str(child.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            entries.append(_build_inventory_entry(child))

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "inventory_roots": [str(root) for root in roots],
        "artifacts": entries,
    }


def _derive_active_names(resolved_resources: Dict[str, Dict[str, Optional[str]]]) -> set[str]:
    names: set[str] = set()
    for resource_info in resolved_resources.values():
        source_path = resource_info.get("source_prompts")
        index_path = resource_info.get("dimension_subpool_index")
        if source_path:
            names.add(Path(source_path).name)
        if index_path:
            index = Path(index_path)
            names.add(index.name)
            names.add(index.parent.name)
    return names


def _derive_fallback_names(
    resource_candidates: taxonomy_tools.RuntimeResourceCandidates,
    resolved_resources: Dict[str, Dict[str, Optional[str]]],
) -> set[str]:
    active_names = _derive_active_names(resolved_resources)
    candidates = {
        resource_candidates.base_source_prompts.name,
        resource_candidates.base_dimension_subpool_index.parent.name,
        resource_candidates.base_cleaned_dimension_subpool_index.parent.name,
        resource_candidates.screened_cleaned_dimension_subpool_index_v1.parent.name,
        resource_candidates.screened_dimension_subpool_index.parent.name,
        resource_candidates.sd35_turbo_dimension_subpool_index_v1.parent.name,
    }
    return {name for name in candidates if name and name not in active_names}


def classify_pool_artifacts(
    *,
    inventory: Dict[str, Any],
    resolved_resources: Dict[str, Dict[str, Optional[str]]],
    resource_candidates: taxonomy_tools.RuntimeResourceCandidates = DEFAULT_RUNTIME_RESOURCE_CANDIDATES,
    stale_candidate_names: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    active_names = _derive_active_names(resolved_resources)
    fallback_names = _derive_fallback_names(resource_candidates, resolved_resources)
    stale_names = set(stale_candidate_names or DEFAULT_STALE_CANDIDATE_NAMES)

    classifications: Dict[str, List[Dict[str, Any]]] = {
        "active": [],
        "fallback": [],
        "stale_candidate": [],
        "historical_candidate": [],
        "unclassified": [],
    }

    for entry in inventory.get("artifacts", []):
        name = entry.get("name", "")
        if name in active_names:
            classifications["active"].append(entry)
        elif name in fallback_names:
            classifications["fallback"].append(entry)
        elif name in stale_names:
            classifications["stale_candidate"].append(entry)
        elif name == "anatomy_screened_dimension_subpools_v2":
            classifications["historical_candidate"].append(entry)
        elif "v1" in name or "v2" in name or "full" in name or "round" in name:
            classifications["historical_candidate"].append(entry)
        else:
            classifications["unclassified"].append(entry)

    return classifications


def load_prompt_pool_routing_config(
    routing_config_path: Path = DEFAULT_ROUTING_CONFIG_PATH,
) -> Dict[str, Any]:
    payload = _read_json(Path(routing_config_path))
    if "routing" not in payload or not isinstance(payload["routing"], dict):
        raise ValueError("prompt_pool_routing_v1.json must contain a top-level 'routing' object")
    return payload


def load_prompt_pool_screening_spec(
    screening_spec_path: Path = DEFAULT_SCREENING_SPEC_PATH,
) -> Dict[str, Any]:
    payload = _read_json(Path(screening_spec_path))
    if "shared_pool_families" not in payload or "dimension_overrides" not in payload:
        raise ValueError(
            "prompt_pool_screening_spec_v1.json must contain 'shared_pool_families' and 'dimension_overrides'"
        )
    return payload


def load_prompt_pool_build_targets(
    build_targets_path: Path = DEFAULT_BUILD_TARGETS_PATH,
) -> Dict[str, Any]:
    payload = _read_json(Path(build_targets_path))
    if "shared_pool_family_outputs" not in payload or "dimension_subpool_outputs" not in payload:
        raise ValueError(
            "prompt_pool_build_targets_v1.json must contain 'shared_pool_family_outputs' and 'dimension_subpool_outputs'"
        )
    return payload


def _current_pool_status_for_dimension(
    *,
    dimension: str,
    coverage_by_model: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    for model_id, coverage in coverage_by_model.items():
        dimension_counts = coverage.get("dimension_counts", {})
        if dimension in dimension_counts:
            status[model_id] = {
                "pool_count": int(dimension_counts[dimension]),
                "current_mode": "active_index_subpool",
            }
        else:
            status[model_id] = {
                "pool_count": None,
                "current_mode": "fallback_runtime_filter",
            }
    return status


def build_prompt_pool_routing(
    *,
    taxonomy_summary: Dict[str, Any],
    routing_config: Dict[str, Any],
    coverage_by_model: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    all_dimensions = list(taxonomy_summary.get("all_dimensions", []))
    config_dimensions = routing_config.get("routing", {})

    output_dimensions: Dict[str, Any] = {}
    missing_in_config: List[str] = []
    for dimension in all_dimensions:
        route = config_dimensions.get(dimension)
        if route is None:
            missing_in_config.append(dimension)
            route = {
                "strategy": "global_cleaned_pool",
                "notes": "未显式配置，暂时回退为全局 cleaned pool。",
            }

        current_status_by_model = _current_pool_status_for_dimension(
            dimension=dimension,
            coverage_by_model=coverage_by_model,
        )
        output_dimensions[dimension] = {
            "strategy": route.get("strategy"),
            "notes": route.get("notes", ""),
            "base_tags": list(route.get("base_tags", [])),
            "shared_pool_family": route.get("shared_pool_family"),
            "llm_screening_required": bool(route.get("llm_screening_required", False)),
            "turbo_pool_strategy": route.get("turbo_pool_strategy"),
            "current_status_by_model": current_status_by_model,
        }

    extra_in_config = sorted(set(config_dimensions.keys()) - set(all_dimensions))

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "version": routing_config.get("version", "v1"),
        "taxonomy_name": taxonomy_summary.get("taxonomy_name"),
        "dimensions": output_dimensions,
        "warnings": {
            "missing_in_config": sorted(missing_in_config),
            "extra_in_config": extra_in_config,
        },
    }


def build_prompt_pool_screening_plan(
    *,
    routing_plan: Dict[str, Any],
) -> Dict[str, Any]:
    dimensions: Dict[str, Any] = {}
    shared_pool_families: Dict[str, Dict[str, Any]] = {}
    for dimension, route in routing_plan.get("dimensions", {}).items():
        if route.get("strategy") != "rule_recall_then_llm_screen":
            continue
        family = route.get("shared_pool_family")
        dimensions[dimension] = {
            "requires_llm_screen": True,
            "rule_recall_tags": list(route.get("base_tags", [])),
            "shared_pool_family": family,
            "target_common_pool_directory": "targeted_dimension_subpools_cleaned_v2",
            "target_turbo_pool_directory": "sd35_turbo_targeted_dimension_subpools_clipsafe_v2",
            "turbo_pool_strategy": route.get("turbo_pool_strategy"),
            "notes": route.get("notes", ""),
        }
        if family:
            family_entry = shared_pool_families.setdefault(
                family,
                {
                    "dimensions": [],
                    "union_rule_recall_tags": [],
                },
            )
            family_entry["dimensions"].append(dimension)
            for tag in route.get("base_tags", []):
                if tag not in family_entry["union_rule_recall_tags"]:
                    family_entry["union_rule_recall_tags"].append(tag)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "dimensions": dimensions,
        "shared_pool_families": shared_pool_families,
    }


def build_prompt_pool_screening_spec(
    *,
    routing_plan: Dict[str, Any],
    screening_spec_config: Dict[str, Any],
) -> Dict[str, Any]:
    shared_families_cfg = dict(screening_spec_config.get("shared_pool_families", {}))
    dimension_overrides = dict(screening_spec_config.get("dimension_overrides", {}))

    dimensions: Dict[str, Any] = {}
    for dimension, route in routing_plan.get("dimensions", {}).items():
        override = dict(dimension_overrides.get(dimension, {}))
        builder_mode = override.get("builder_mode")
        if builder_mode is None:
            builder_mode = (
                "rule_recall_then_llm_screen"
                if route.get("strategy") == "rule_recall_then_llm_screen"
                else "rule_recall_only"
            )
        requires_llm_screen = override.get("requires_llm_screen")
        if requires_llm_screen is None:
            requires_llm_screen = bool(route.get("llm_screening_required", False))

        dimensions[dimension] = {
            "shared_pool_family": override.get("shared_pool_family", route.get("shared_pool_family")),
            "builder_mode": builder_mode,
            "requires_llm_screen": bool(requires_llm_screen),
            "base_tags": list(override.get("base_tags", route.get("base_tags", []))),
            "turbo_pool_strategy": override.get("turbo_pool_strategy", route.get("turbo_pool_strategy")),
            "dimension_focus": list(override.get("dimension_focus", [])),
            "notes": route.get("notes", ""),
        }

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "version": screening_spec_config.get("version", "v1"),
        "shared_pool_families": shared_families_cfg,
        "dimensions": dimensions,
    }


def build_prompt_pool_build_targets(
    *,
    screening_spec: Dict[str, Any],
    build_targets_config: Dict[str, Any],
) -> Dict[str, Any]:
    dimensions: Dict[str, Any] = {}
    for dimension, spec in screening_spec.get("dimensions", {}).items():
        dimensions[dimension] = {
            "build_mode": spec.get("builder_mode"),
            "shared_pool_family": spec.get("shared_pool_family"),
            "turbo_pool_strategy": spec.get("turbo_pool_strategy"),
            "filename": f"{dimension}.jsonl",
        }

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "version": build_targets_config.get("version", "v1"),
        "shared_pool_family_outputs": build_targets_config.get("shared_pool_family_outputs", {}),
        "dimension_subpool_outputs": build_targets_config.get("dimension_subpool_outputs", {}),
        "dimensions": dimensions,
        "notes": build_targets_config.get("notes", []),
    }


def build_cleanup_candidates(
    *,
    classifications: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "active": list(classifications.get("active", [])),
        "fallback": list(classifications.get("fallback", [])),
        "stale_candidate": list(classifications.get("stale_candidate", [])),
        "historical_candidate": list(classifications.get("historical_candidate", [])),
        "delete_requires_confirmation": True,
        "notes": [
            "本文件只输出候选清单，不执行任何外部 pool 删除。",
            "只有在人工确认后，才允许清理 stale_candidate / historical_candidate。",
        ],
    }


__all__ = [
    "DEFAULT_MODEL_IDS",
    "DEFAULT_POOL_INVENTORY_ROOTS",
    "DEFAULT_BUILD_TARGETS_PATH",
    "DEFAULT_ROUTING_CONFIG_PATH",
    "DEFAULT_SCREENING_SPEC_PATH",
    "DEFAULT_RUNTIME_RESOURCE_CANDIDATES",
    "DEFAULT_STALE_CANDIDATE_NAMES",
    "build_cleanup_candidates",
    "build_prompt_pool_build_targets",
    "build_prompt_pool_routing",
    "build_prompt_pool_screening_plan",
    "build_prompt_pool_screening_spec",
    "classify_pool_artifacts",
    "inspect_active_pool_coverage",
    "load_prompt_pool_build_targets",
    "load_prompt_pool_routing_config",
    "load_prompt_pool_screening_spec",
    "resolve_active_prompt_pools",
    "scan_prompt_pool_inventory",
]
