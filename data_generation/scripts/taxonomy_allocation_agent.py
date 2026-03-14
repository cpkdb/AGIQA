#!/usr/bin/env python3
"""
Taxonomy & Allocation Agent

First version:
- deterministic
- offline analysis only
- no automatic policy decisions
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import taxonomy_allocation_tools as tools


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _write_text(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def run_taxonomy_allocation_agent(
    *,
    output_dir: Path,
    runs_root: Path = tools.DEFAULT_RUNS_ROOT,
    taxonomy_path: Path = tools.DEFAULT_TAXONOMY_PATH,
    resource_candidates: tools.RuntimeResourceCandidates = tools.DEFAULT_RUNTIME_RESOURCE_CANDIDATES,
    model_ids: Optional[Sequence[str]] = None,
    recent_run_limit: Optional[int] = None,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    taxonomy_summary = tools.load_taxonomy_summary(Path(taxonomy_path))
    resolved_resources = tools.resolve_runtime_resources(
        resource_candidates=resource_candidates,
        model_ids=model_ids,
    )

    coverage_by_model: Dict[str, Dict[str, Any]] = {}
    for model_id, resource_info in resolved_resources.items():
        source_prompts = resource_info.get("source_prompts")
        dimension_subpool_index = resource_info.get("dimension_subpool_index")
        if source_prompts is None or dimension_subpool_index is None:
            coverage_by_model[model_id] = {
                "source_prompts": source_prompts,
                "dimension_subpool_index": dimension_subpool_index,
                "total_source_prompts": 0,
                "dimension_counts": {},
                "warnings": {
                    "low_pool_dims": [],
                    "missing_dims": list(taxonomy_summary.get("all_dimensions", [])),
                },
            }
            continue

        coverage_by_model[model_id] = tools.inspect_pool_coverage(
            source_prompts_path=Path(source_prompts),
            dimension_subpool_index=Path(dimension_subpool_index),
            dimensions=taxonomy_summary.get("all_dimensions"),
        )

    model_dimension_stats = tools.aggregate_run_statistics(
        runs_root=Path(runs_root),
        model_filter=model_ids,
        recent_run_limit=recent_run_limit,
    )
    speed_summary = tools.summarize_generation_speed(
        runs_root=Path(runs_root),
        model_filter=model_ids,
        recent_run_limit=recent_run_limit,
    )

    coverage_summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "taxonomy": {
            "path": str(Path(taxonomy_path)),
            "name": taxonomy_summary.get("taxonomy_name"),
            "total_dimensions": taxonomy_summary.get("total_dimensions"),
            "perspectives": taxonomy_summary.get("perspectives"),
        },
        "runtime_resources": resolved_resources,
        "coverage_by_model": coverage_by_model,
    }

    allocation_plan_template = tools.build_allocation_plan_template(
        taxonomy_summary=taxonomy_summary,
        coverage_by_model=coverage_by_model,
        model_dimension_stats=model_dimension_stats,
        speed_summary=speed_summary,
    )
    allocation_insights = tools.build_allocation_insights_markdown(
        taxonomy_summary=taxonomy_summary,
        resolved_resources=resolved_resources,
        coverage_by_model=coverage_by_model,
        model_dimension_stats=model_dimension_stats,
        speed_summary=speed_summary,
    )

    artifact_paths = {
        "coverage_summary": output_root / "coverage_summary.json",
        "model_dimension_stats": output_root / "model_dimension_stats.json",
        "speed_summary": output_root / "speed_summary.json",
        "allocation_plan_template": output_root / "allocation_plan.template.json",
        "allocation_insights": output_root / "allocation_insights.md",
        "manifest": output_root / "manifest.json",
    }

    _write_json(artifact_paths["coverage_summary"], coverage_summary)
    _write_json(artifact_paths["model_dimension_stats"], model_dimension_stats)
    _write_json(artifact_paths["speed_summary"], speed_summary)
    _write_json(artifact_paths["allocation_plan_template"], allocation_plan_template)
    _write_text(artifact_paths["allocation_insights"], allocation_insights)

    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "agent": "taxonomy_allocation_agent",
        "taxonomy_path": str(Path(taxonomy_path)),
        "runs_root": str(Path(runs_root)),
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }
    _write_json(artifact_paths["manifest"], manifest)

    return {
        "generated_at": manifest["generated_at"],
        "artifacts": manifest["artifacts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Taxonomy & Allocation Agent (analysis-only v1).")
    parser.add_argument("--output_dir", required=True, help="Directory to write agent artifacts into.")
    parser.add_argument("--runs_root", default=str(tools.DEFAULT_RUNS_ROOT), help="Root directory containing historical run outputs.")
    parser.add_argument("--taxonomy_path", default=str(tools.DEFAULT_TAXONOMY_PATH), help="Active taxonomy JSON path.")
    parser.add_argument("--model_filter", default="", help="Optional comma-separated model ids.")
    parser.add_argument("--recent_run_limit", type=int, default=None, help="Optional limit on the number of newest runs to inspect.")
    args = parser.parse_args()

    model_filter = [value.strip() for value in args.model_filter.split(",") if value.strip()] or None
    result = run_taxonomy_allocation_agent(
        output_dir=Path(args.output_dir),
        runs_root=Path(args.runs_root),
        taxonomy_path=Path(args.taxonomy_path),
        model_ids=model_filter,
        recent_run_limit=args.recent_run_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
