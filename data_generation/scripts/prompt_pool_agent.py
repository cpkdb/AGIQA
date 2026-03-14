#!/usr/bin/env python3
"""
Prompt Pool Agent

First version:
- deterministic
- offline analysis only
- no deletion
- no new pool building
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

import prompt_pool_agent_tools as tools
import taxonomy_allocation_tools as taxonomy_tools


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def run_prompt_pool_agent(
    *,
    output_dir: Path,
    taxonomy_path: Path = tools.DEFAULT_TAXONOMY_PATH,
    routing_config_path: Path = tools.DEFAULT_ROUTING_CONFIG_PATH,
    screening_spec_path: Path = tools.DEFAULT_SCREENING_SPEC_PATH,
    build_targets_path: Path = tools.DEFAULT_BUILD_TARGETS_PATH,
    resource_candidates: taxonomy_tools.RuntimeResourceCandidates = tools.DEFAULT_RUNTIME_RESOURCE_CANDIDATES,
    inventory_roots: Optional[Sequence[Path]] = None,
    model_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    taxonomy_summary = taxonomy_tools.load_taxonomy_summary(Path(taxonomy_path))
    resolved_resources = tools.resolve_active_prompt_pools(
        resource_candidates=resource_candidates,
        model_ids=model_ids,
    )
    coverage_by_model = tools.inspect_active_pool_coverage(
        resolved_resources=resolved_resources,
        dimensions=taxonomy_summary.get("all_dimensions", []),
    )

    active_pool_manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "agent": "prompt_pool_agent",
        "taxonomy": {
            "path": str(Path(taxonomy_path)),
            "name": taxonomy_summary.get("taxonomy_name"),
            "total_dimensions": taxonomy_summary.get("total_dimensions"),
        },
        "runtime_resources": resolved_resources,
        "coverage_by_model": coverage_by_model,
    }

    inventory = tools.scan_prompt_pool_inventory(inventory_roots=inventory_roots)
    classifications = tools.classify_pool_artifacts(
        inventory=inventory,
        resolved_resources=resolved_resources,
        resource_candidates=resource_candidates,
    )
    inventory_payload = {
        **inventory,
        "classifications": classifications,
    }

    routing_config = tools.load_prompt_pool_routing_config(Path(routing_config_path))
    routing_payload = tools.build_prompt_pool_routing(
        taxonomy_summary=taxonomy_summary,
        routing_config=routing_config,
        coverage_by_model=coverage_by_model,
    )
    screening_plan = tools.build_prompt_pool_screening_plan(routing_plan=routing_payload)
    screening_spec_config = tools.load_prompt_pool_screening_spec(Path(screening_spec_path))
    screening_spec = tools.build_prompt_pool_screening_spec(
        routing_plan=routing_payload,
        screening_spec_config=screening_spec_config,
    )
    build_targets_config = tools.load_prompt_pool_build_targets(Path(build_targets_path))
    build_targets = tools.build_prompt_pool_build_targets(
        screening_spec=screening_spec,
        build_targets_config=build_targets_config,
    )
    cleanup_candidates = tools.build_cleanup_candidates(classifications=classifications)

    artifact_paths = {
        "active_pool_manifest": output_root / "active_pool_manifest.json",
        "prompt_pool_inventory": output_root / "prompt_pool_inventory.json",
        "prompt_pool_routing": output_root / "prompt_pool_routing_v1.json",
        "prompt_pool_screening_plan": output_root / "prompt_pool_screening_plan_v1.json",
        "prompt_pool_screening_spec": output_root / "prompt_pool_screening_spec_v1.json",
        "prompt_pool_build_targets": output_root / "prompt_pool_build_targets_v1.json",
        "prompt_pool_cleanup_candidates": output_root / "prompt_pool_cleanup_candidates_v1.json",
        "manifest": output_root / "manifest.json",
    }

    _write_json(artifact_paths["active_pool_manifest"], active_pool_manifest)
    _write_json(artifact_paths["prompt_pool_inventory"], inventory_payload)
    _write_json(artifact_paths["prompt_pool_routing"], routing_payload)
    _write_json(artifact_paths["prompt_pool_screening_plan"], screening_plan)
    _write_json(artifact_paths["prompt_pool_screening_spec"], screening_spec)
    _write_json(artifact_paths["prompt_pool_build_targets"], build_targets)
    _write_json(artifact_paths["prompt_pool_cleanup_candidates"], cleanup_candidates)

    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "agent": "prompt_pool_agent",
        "taxonomy_path": str(Path(taxonomy_path)),
        "routing_config_path": str(Path(routing_config_path)),
        "screening_spec_path": str(Path(screening_spec_path)),
        "build_targets_path": str(Path(build_targets_path)),
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }
    _write_json(artifact_paths["manifest"], manifest)

    return {
        "generated_at": manifest["generated_at"],
        "artifacts": manifest["artifacts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Prompt Pool Agent (analysis-only v1).")
    parser.add_argument("--output_dir", required=True, help="Directory to write agent artifacts into.")
    parser.add_argument(
        "--taxonomy_path",
        default=str(tools.DEFAULT_TAXONOMY_PATH),
        help="Active taxonomy JSON path.",
    )
    parser.add_argument(
        "--routing_config_path",
        default=str(tools.DEFAULT_ROUTING_CONFIG_PATH),
        help="Prompt pool routing config path.",
    )
    parser.add_argument(
        "--screening_spec_path",
        default=str(tools.DEFAULT_SCREENING_SPEC_PATH),
        help="Prompt pool screening spec config path.",
    )
    parser.add_argument(
        "--build_targets_path",
        default=str(tools.DEFAULT_BUILD_TARGETS_PATH),
        help="Prompt pool build-target naming config path.",
    )
    parser.add_argument(
        "--inventory_roots",
        default="",
        help="Optional comma-separated inventory roots to scan.",
    )
    parser.add_argument("--model_filter", default="", help="Optional comma-separated model ids.")
    args = parser.parse_args()

    model_filter = [value.strip() for value in args.model_filter.split(",") if value.strip()] or None
    inventory_roots = [Path(value.strip()) for value in args.inventory_roots.split(",") if value.strip()] or None
    result = run_prompt_pool_agent(
        output_dir=Path(args.output_dir),
        taxonomy_path=Path(args.taxonomy_path),
        routing_config_path=Path(args.routing_config_path),
        screening_spec_path=Path(args.screening_spec_path),
        build_targets_path=Path(args.build_targets_path),
        inventory_roots=inventory_roots,
        model_ids=model_filter,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
