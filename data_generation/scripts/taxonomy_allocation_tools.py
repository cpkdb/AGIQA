#!/usr/bin/env python3
"""
Deterministic tools for the Taxonomy & Allocation Agent.

The first version is analysis-only:
- resolve current runtime prompt resources
- summarize active taxonomy
- inspect pool coverage
- aggregate historical run success/failure statistics
- estimate coarse model speeds from historical run artifacts
- build an editable allocation-plan template
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY_PATH = SCRIPT_DIR.parent / "config" / "quality_dimensions_active.json"
DEFAULT_RUNS_ROOT = Path("/root/autodl-tmp")

DEFAULT_MODEL_IDS = [
    "flux-schnell",
    "sd3.5-large-turbo",
    "qwen-image-lightning",
]


class RuntimeResourceCandidates:
    def __init__(
        self,
        *,
        base_source_prompts: Path,
        cleaned_source_prompts: Path,
        sd35_turbo_source_prompts: Path,
        base_dimension_subpool_index: Path,
        base_cleaned_dimension_subpool_index: Path,
        semantic_screened_dimension_subpool_index_v1: Path,
        screened_cleaned_dimension_subpool_index_v2: Path,
        screened_cleaned_dimension_subpool_index_v1: Path,
        screened_dimension_subpool_index: Path,
        sd35_turbo_semantic_screened_dimension_subpool_index_v1: Path,
        sd35_turbo_dimension_subpool_index_v2: Path,
        sd35_turbo_dimension_subpool_index_v1: Path,
    ) -> None:
        self.base_source_prompts = Path(base_source_prompts)
        self.cleaned_source_prompts = Path(cleaned_source_prompts)
        self.sd35_turbo_source_prompts = Path(sd35_turbo_source_prompts)
        self.base_dimension_subpool_index = Path(base_dimension_subpool_index)
        self.base_cleaned_dimension_subpool_index = Path(base_cleaned_dimension_subpool_index)
        self.semantic_screened_dimension_subpool_index_v1 = Path(semantic_screened_dimension_subpool_index_v1)
        self.screened_cleaned_dimension_subpool_index_v2 = Path(screened_cleaned_dimension_subpool_index_v2)
        self.screened_cleaned_dimension_subpool_index_v1 = Path(screened_cleaned_dimension_subpool_index_v1)
        self.screened_dimension_subpool_index = Path(screened_dimension_subpool_index)
        self.sd35_turbo_semantic_screened_dimension_subpool_index_v1 = Path(
            sd35_turbo_semantic_screened_dimension_subpool_index_v1
        )
        self.sd35_turbo_dimension_subpool_index_v2 = Path(sd35_turbo_dimension_subpool_index_v2)
        self.sd35_turbo_dimension_subpool_index_v1 = Path(sd35_turbo_dimension_subpool_index_v1)


DEFAULT_RUNTIME_RESOURCE_CANDIDATES = RuntimeResourceCandidates(
    base_source_prompts=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool.jsonl"
    ),
    cleaned_source_prompts=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl"
    ),
    sd35_turbo_source_prompts=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_sd35_turbo_clipsafe_v1.jsonl"
    ),
    base_dimension_subpool_index=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/dimension_subpools/index.json"
    ),
    base_cleaned_dimension_subpool_index=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/dimension_subpools_cleaned_v1/index.json"
    ),
    semantic_screened_dimension_subpool_index_v1=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/semantic_screened_dimension_subpools_cleaned_v1/index.json"
    ),
    screened_cleaned_dimension_subpool_index_v2=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v2/index.json"
    ),
    screened_cleaned_dimension_subpool_index_v1=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v1/index.json"
    ),
    screened_dimension_subpool_index=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools/index.json"
    ),
    sd35_turbo_semantic_screened_dimension_subpool_index_v1=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1/index.json"
    ),
    sd35_turbo_dimension_subpool_index_v2=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/sd35_turbo_dimension_subpools_clipsafe_v2/index.json"
    ),
    sd35_turbo_dimension_subpool_index_v1=Path(
        "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/sd35_turbo_dimension_subpools_clipsafe_v1/index.json"
    ),
)


def _path_or_none(path: Optional[Path]) -> Optional[str]:
    return str(path) if path is not None else None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count_prompt_rows(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix.lower() == ".jsonl":
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    payload = _read_json(path)
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
        return len(payload["prompts"])
    return 0


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _resolve_shared_source_prompts(resource_candidates: RuntimeResourceCandidates) -> Path:
    if resource_candidates.cleaned_source_prompts.exists():
        return resource_candidates.cleaned_source_prompts
    return resource_candidates.base_source_prompts


def _resolve_shared_dimension_subpool_index(resource_candidates: RuntimeResourceCandidates) -> Path:
    if resource_candidates.semantic_screened_dimension_subpool_index_v1.exists():
        return resource_candidates.semantic_screened_dimension_subpool_index_v1
    if resource_candidates.screened_cleaned_dimension_subpool_index_v2.exists():
        return resource_candidates.screened_cleaned_dimension_subpool_index_v2
    if resource_candidates.screened_cleaned_dimension_subpool_index_v1.exists():
        return resource_candidates.screened_cleaned_dimension_subpool_index_v1
    if resource_candidates.base_cleaned_dimension_subpool_index.exists():
        return resource_candidates.base_cleaned_dimension_subpool_index
    if resource_candidates.screened_dimension_subpool_index.exists():
        return resource_candidates.screened_dimension_subpool_index
    return resource_candidates.base_dimension_subpool_index


def resolve_runtime_resources(
    *,
    resource_candidates: RuntimeResourceCandidates = DEFAULT_RUNTIME_RESOURCE_CANDIDATES,
    model_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    selected_models = list(model_ids) if model_ids else list(DEFAULT_MODEL_IDS)

    shared_source = _resolve_shared_source_prompts(resource_candidates)
    shared_index = _resolve_shared_dimension_subpool_index(resource_candidates)

    resolved: Dict[str, Dict[str, Optional[str]]] = {}
    for model_id in selected_models:
        source_prompts = shared_source
        dimension_subpool_index = shared_index

        if model_id == "sd3.5-large-turbo":
            if resource_candidates.sd35_turbo_source_prompts.exists():
                source_prompts = resource_candidates.sd35_turbo_source_prompts
            if resource_candidates.sd35_turbo_semantic_screened_dimension_subpool_index_v1.exists():
                dimension_subpool_index = resource_candidates.sd35_turbo_semantic_screened_dimension_subpool_index_v1
            elif resource_candidates.sd35_turbo_dimension_subpool_index_v2.exists():
                dimension_subpool_index = resource_candidates.sd35_turbo_dimension_subpool_index_v2
            elif resource_candidates.sd35_turbo_dimension_subpool_index_v1.exists():
                dimension_subpool_index = resource_candidates.sd35_turbo_dimension_subpool_index_v1

        resolved[model_id] = {
            "source_prompts": _path_or_none(source_prompts),
            "dimension_subpool_index": _path_or_none(dimension_subpool_index),
        }

    return resolved


def load_taxonomy_summary(taxonomy_path: Path = DEFAULT_TAXONOMY_PATH) -> Dict[str, Any]:
    payload = _read_json(Path(taxonomy_path))
    perspectives = payload.get("perspectives", {})
    perspective_dimensions: Dict[str, List[str]] = {}
    all_dimensions: List[str] = []

    for perspective_name, perspective_payload in perspectives.items():
        dimensions = list((perspective_payload or {}).get("dimensions", {}).keys())
        perspective_dimensions[perspective_name] = dimensions
        all_dimensions.extend(dimensions)

    return {
        "taxonomy_name": payload.get("taxonomy_name"),
        "taxonomy_path": str(Path(taxonomy_path)),
        "total_dimensions": payload.get("statistics", {}).get("total_dimensions", len(all_dimensions)),
        "perspectives": perspective_dimensions,
        "all_dimensions": all_dimensions,
    }


def inspect_pool_coverage(
    *,
    source_prompts_path: Path,
    dimension_subpool_index: Path,
    dimensions: Optional[Sequence[str]] = None,
    low_pool_threshold: int = 2000,
) -> Dict[str, Any]:
    source_path = Path(source_prompts_path)
    index_path = Path(dimension_subpool_index)
    payload = _read_json(index_path)
    index_dimensions = payload.get("dimensions", {})

    selected_dimensions = list(dimensions) if dimensions else list(index_dimensions.keys())
    dimension_counts: Dict[str, int] = {}
    missing_dims: List[str] = []
    for dimension in selected_dimensions:
        meta = index_dimensions.get(dimension)
        if not isinstance(meta, dict):
            missing_dims.append(dimension)
            continue
        dimension_counts[dimension] = int(meta.get("count", 0))

    low_pool_dims = sorted(
        [dimension for dimension, count in dimension_counts.items() if count < low_pool_threshold]
    )

    return {
        "source_prompts": str(source_path),
        "dimension_subpool_index": str(index_path),
        "total_source_prompts": _count_prompt_rows(source_path),
        "dimension_counts": dimension_counts,
        "warnings": {
            "low_pool_dims": low_pool_dims,
            "missing_dims": sorted(missing_dims),
        },
    }


def _discover_run_directories(
    runs_root: Path,
    recent_run_limit: Optional[int] = None,
) -> List[Path]:
    run_dirs: List[Path] = []
    for report_path in Path(runs_root).rglob("validation_report.json"):
        run_dir = report_path.parent
        if (run_dir / "dataset.json").exists() and (run_dir / "full_log.json").exists():
            run_dirs.append(run_dir)

    def sort_key(path: Path) -> float:
        dataset_payload = _read_json(path / "dataset.json")
        completed_at = _parse_iso8601(dataset_payload.get("metadata", {}).get("completed_at"))
        if completed_at is not None:
            return completed_at.timestamp()
        return (path / "validation_report.json").stat().st_mtime

    run_dirs = sorted(run_dirs, key=sort_key, reverse=True)
    if recent_run_limit is not None:
        run_dirs = run_dirs[:recent_run_limit]
    return run_dirs


def _normalize_filter(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    return {value for value in values if value}


def aggregate_run_statistics(
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    model_filter: Optional[Sequence[str]] = None,
    dimension_filter: Optional[Sequence[str]] = None,
    recent_run_limit: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    allowed_models = _normalize_filter(model_filter)
    allowed_dimensions = _normalize_filter(dimension_filter)

    stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {
            "pairs": 0,
            "valid_pairs": 0,
            "invalid_pairs": 0,
            "total_attempts": 0,
            "avg_attempts": 0.0,
            "validation_rate": 0.0,
            "failure_types": {},
        }))
    )

    for run_dir in _discover_run_directories(Path(runs_root), recent_run_limit=recent_run_limit):
        dataset_payload = _read_json(run_dir / "dataset.json")
        model_id = dataset_payload.get("metadata", {}).get("model_id", "unknown")
        if allowed_models and model_id not in allowed_models:
            continue

        full_log = _read_json(run_dir / "full_log.json")
        for pair in full_log:
            dimension = pair.get("dimension")
            severity = pair.get("severity")
            if not dimension or not severity:
                continue
            if allowed_dimensions and dimension not in allowed_dimensions:
                continue

            bucket = stats[model_id][dimension][severity]
            bucket["pairs"] += 1
            attempts = int(pair.get("total_attempts", len(pair.get("attempts", [])) or 0))
            bucket["total_attempts"] += attempts

            if pair.get("success"):
                bucket["valid_pairs"] += 1
            else:
                bucket["invalid_pairs"] += 1
                failure = (
                    pair.get("final_validation", {}).get("failure")
                    or "unknown"
                )
                failure_types = bucket["failure_types"]
                failure_types[failure] = int(failure_types.get(failure, 0)) + 1

    materialized: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model_id, dimensions in stats.items():
        materialized[model_id] = {}
        for dimension, severities in dimensions.items():
            materialized[model_id][dimension] = {}
            for severity, bucket in severities.items():
                pairs = bucket["pairs"]
                bucket["avg_attempts"] = round(bucket["total_attempts"] / pairs, 2) if pairs else 0.0
                bucket["validation_rate"] = round(bucket["valid_pairs"] / pairs, 4) if pairs else 0.0
                materialized[model_id][dimension][severity] = dict(bucket)

    return materialized


def summarize_generation_speed(
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    model_filter: Optional[Sequence[str]] = None,
    recent_run_limit: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    allowed_models = _normalize_filter(model_filter)
    aggregates: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "total_elapsed_seconds": 0.0,
        "total_pairs": 0.0,
        "total_valid_pairs": 0.0,
        "run_count": 0.0,
    })

    for run_dir in _discover_run_directories(Path(runs_root), recent_run_limit=recent_run_limit):
        dataset_payload = _read_json(run_dir / "dataset.json")
        model_id = dataset_payload.get("metadata", {}).get("model_id", "unknown")
        if allowed_models and model_id not in allowed_models:
            continue

        created_at = _parse_iso8601(dataset_payload.get("metadata", {}).get("created_at"))
        completed_at = _parse_iso8601(dataset_payload.get("metadata", {}).get("completed_at"))
        if created_at is None or completed_at is None or completed_at <= created_at:
            continue

        validation_payload = _read_json(run_dir / "validation_report.json")
        summary = validation_payload.get("summary", {})
        total_pairs = float(summary.get("total_pairs", 0))
        valid_pairs = float(summary.get("valid_pairs", 0))
        if total_pairs <= 0:
            continue

        elapsed_seconds = (completed_at - created_at).total_seconds()
        bucket = aggregates[model_id]
        bucket["total_elapsed_seconds"] += elapsed_seconds
        bucket["total_pairs"] += total_pairs
        bucket["total_valid_pairs"] += valid_pairs
        bucket["run_count"] += 1

    materialized: Dict[str, Dict[str, Any]] = {}
    for model_id, bucket in aggregates.items():
        total_elapsed = bucket["total_elapsed_seconds"]
        total_pairs = bucket["total_pairs"]
        total_valid_pairs = bucket["total_valid_pairs"]
        materialized[model_id] = {
            "avg_pair_seconds": round(total_elapsed / total_pairs, 2) if total_pairs else None,
            "avg_success_pair_seconds": round(total_elapsed / total_valid_pairs, 2) if total_valid_pairs else None,
            "avg_image_seconds": round(total_elapsed / (total_pairs * 2), 2) if total_pairs else None,
            "run_count": int(bucket["run_count"]),
            "notes": "Estimated from dataset created_at/completed_at and validation_report total_pairs; includes retry/judge overhead.",
        }

    return materialized


def build_allocation_plan_template(
    *,
    taxonomy_summary: Dict[str, Any],
    coverage_by_model: Dict[str, Dict[str, Any]],
    model_dimension_stats: Dict[str, Dict[str, Dict[str, Any]]],
    speed_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    dimension_entries: Dict[str, Any] = {}

    for dimension in taxonomy_summary.get("all_dimensions", []):
        historical_success: Dict[str, Optional[float]] = {}
        pool_size_by_model: Dict[str, Optional[int]] = {}

        for model_id in DEFAULT_MODEL_IDS:
            pool_size_by_model[model_id] = coverage_by_model.get(model_id, {}).get("dimension_counts", {}).get(dimension)

            model_stats = model_dimension_stats.get(model_id, {}).get(dimension, {})
            total_pairs = sum(bucket.get("pairs", 0) for bucket in model_stats.values())
            total_valid_pairs = sum(bucket.get("valid_pairs", 0) for bucket in model_stats.values())
            historical_success[model_id] = round(total_valid_pairs / total_pairs, 4) if total_pairs else None

        dimension_entries[dimension] = {
            "pool_size_by_model": pool_size_by_model,
            "historical_success": historical_success,
            "policy": {
                "sampling_multiplier": None,
                "allowed_models": None,
                "allowed_severities": None,
                "judge_mode": None,
                "notes": "",
            },
        }

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "defaults": {
            "sampling_multiplier": 1.0,
            "allowed_models": list(DEFAULT_MODEL_IDS),
            "allowed_severities": ["moderate", "severe"],
            "judge_mode": "inherit",
        },
        "context": {
            "taxonomy_name": taxonomy_summary.get("taxonomy_name"),
            "total_dimensions": taxonomy_summary.get("total_dimensions"),
            "speed_summary": speed_summary,
        },
        "dimensions": dimension_entries,
    }


def build_allocation_insights_markdown(
    *,
    taxonomy_summary: Dict[str, Any],
    resolved_resources: Dict[str, Dict[str, Optional[str]]],
    coverage_by_model: Dict[str, Dict[str, Any]],
    model_dimension_stats: Dict[str, Dict[str, Dict[str, Any]]],
    speed_summary: Dict[str, Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Taxonomy & Allocation Insights")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.utcnow().isoformat()}`")
    lines.append(f"- Taxonomy: `{taxonomy_summary.get('taxonomy_name')}`")
    lines.append(f"- Total dimensions: `{taxonomy_summary.get('total_dimensions')}`")
    lines.append("")

    lines.append("## Runtime Resources")
    lines.append("")
    for model_id in DEFAULT_MODEL_IDS:
        if model_id not in resolved_resources:
            continue
        resource_info = resolved_resources[model_id]
        lines.append(f"### {model_id}")
        lines.append(f"- Source prompts: `{resource_info.get('source_prompts')}`")
        lines.append(f"- Dimension subpool index: `{resource_info.get('dimension_subpool_index')}`")
        lines.append("")

    lines.append("## Pool Coverage")
    lines.append("")
    for model_id in DEFAULT_MODEL_IDS:
        coverage = coverage_by_model.get(model_id)
        if not coverage:
            continue
        low_pool_dims = coverage.get("warnings", {}).get("low_pool_dims", [])
        lines.append(f"### {model_id}")
        lines.append(f"- Total source prompts: `{coverage.get('total_source_prompts', 0)}`")
        lines.append(f"- Low-pool dimensions (<2000): `{len(low_pool_dims)}`")
        if low_pool_dims:
            preview = ", ".join(low_pool_dims[:8])
            lines.append(f"- Examples: `{preview}`")
        lines.append("")

    lines.append("## Historical Success")
    lines.append("")
    for model_id in DEFAULT_MODEL_IDS:
        dimension_stats = model_dimension_stats.get(model_id, {})
        if not dimension_stats:
            continue
        summaries: List[tuple[str, float]] = []
        for dimension, severities in dimension_stats.items():
            pairs = sum(bucket.get("pairs", 0) for bucket in severities.values())
            valid_pairs = sum(bucket.get("valid_pairs", 0) for bucket in severities.values())
            if pairs:
                summaries.append((dimension, valid_pairs / pairs))
        summaries.sort(key=lambda item: item[1], reverse=True)

        lines.append(f"### {model_id}")
        if summaries:
            best_preview = ", ".join(f"{name} ({rate:.2f})" for name, rate in summaries[:5])
            worst_preview = ", ".join(f"{name} ({rate:.2f})" for name, rate in summaries[-5:])
            lines.append(f"- Stronger dimensions: `{best_preview}`")
            lines.append(f"- Weaker dimensions: `{worst_preview}`")
        else:
            lines.append("- No historical statistics found.")
        lines.append("")

    lines.append("## Speed Summary")
    lines.append("")
    for model_id in DEFAULT_MODEL_IDS:
        speed = speed_summary.get(model_id)
        if not speed:
            continue
        lines.append(f"### {model_id}")
        lines.append(f"- Avg pair seconds: `{speed.get('avg_pair_seconds')}`")
        lines.append(f"- Avg success pair seconds: `{speed.get('avg_success_pair_seconds')}`")
        lines.append(f"- Avg image seconds: `{speed.get('avg_image_seconds')}`")
        lines.append(f"- Runs considered: `{speed.get('run_count')}`")
        lines.append("")

    lines.append("## Next Step")
    lines.append("")
    lines.append("- Edit `allocation_plan.template.json` into a run-specific `allocation_plan.json` when production policy is ready.")
    lines.append("- Keep this agent analysis-only until model/dimension policies are explicitly decided.")
    lines.append("")

    return "\n".join(lines)


__all__ = [
    "DEFAULT_MODEL_IDS",
    "DEFAULT_RUNTIME_RESOURCE_CANDIDATES",
    "DEFAULT_RUNS_ROOT",
    "DEFAULT_TAXONOMY_PATH",
    "RuntimeResourceCandidates",
    "aggregate_run_statistics",
    "build_allocation_insights_markdown",
    "build_allocation_plan_template",
    "inspect_pool_coverage",
    "load_taxonomy_summary",
    "resolve_runtime_resources",
    "summarize_generation_speed",
]
