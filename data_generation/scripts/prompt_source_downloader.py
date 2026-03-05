#!/usr/bin/env python3
"""
Prompt source downloader manifest utilities.

This module does not fetch remote data yet. It prepares a metadata-only
download manifest based on the validated prompt source plan so that real
download execution can be added in a later step.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, List


def _load_plan_module():
    module_path = Path(__file__).parent / "prompt_source_plan.py"
    spec = spec_from_file_location("prompt_source_plan", module_path)
    module = module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("Failed to load prompt_source_plan.py")
    spec.loader.exec_module(module)
    return module


def build_download_manifest() -> Dict:
    plan_module = _load_plan_module()
    plan = plan_module.build_download_plan()
    jobs: List[Dict] = []

    for source in plan["sources"]:
        jobs.append(
            {
                "source": source["name"],
                "role": source["role"],
                "candidate_pool_size": source["candidate_pool_size"],
                "final_selected_size": source["final_selected_size"],
                "official_urls": source["official_urls"],
                "fields_to_keep": source["fields_to_keep"],
                "download_images": source["download_images"],
                "has_bucket_plan": bool(source.get("buckets")),
            }
        )

    return {
        "mode": "metadata_only",
        "target_final_pool_size": plan["target_final_pool_size"],
        "planned_selected_before_top_up": plan["planned_selected_before_top_up"],
        "planned_top_up_size": plan["planned_top_up_size"],
        "jobs": jobs,
    }


def write_download_manifest(output_dir: str) -> str:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = build_download_manifest()
    manifest_path = output_path / "prompt_download_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return str(manifest_path)


__all__ = [
    "build_download_manifest",
    "write_download_manifest",
]
