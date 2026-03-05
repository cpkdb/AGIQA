#!/usr/bin/env python3
"""
Prompt source plan loader and local workspace preparer.

This module merges the prompt source catalog and sampling spec into a single
validated plan object, then prepares a local directory layout for later
download and stratified sampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_PATH = DATA_DIR / "prompt_source_catalog.json"
SAMPLING_SPEC_PATH = DATA_DIR / "prompt_sampling_spec.json"


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_download_plan() -> Dict:
    catalog = _read_json(CATALOG_PATH)
    sampling_spec = _read_json(SAMPLING_SPEC_PATH)

    source_index = {source["name"]: dict(source) for source in catalog["sources"]}

    for sampled_source in sampling_spec["sources"]:
        name = sampled_source["name"]
        if name not in source_index:
            raise ValueError(f"Sampling spec references unknown source: {name}")

        catalog_source = source_index[name]
        if catalog_source["candidate_pool_size"] != sampled_source["candidate_pool_size"]:
            raise ValueError(f"Candidate pool mismatch for source: {name}")
        if catalog_source["final_selected_size"] != sampled_source["final_selected_size"]:
            raise ValueError(f"Final selected mismatch for source: {name}")

        bucket_total = sum(bucket["final_selected_size"] for bucket in sampled_source["buckets"])
        if bucket_total != sampled_source["final_selected_size"]:
            raise ValueError(f"Bucket totals do not match final selected size for source: {name}")

        catalog_source["buckets"] = sampled_source["buckets"]

    planned_selected_before_top_up = sum(
        source["final_selected_size"] for source in catalog["sources"]
    )

    return {
        "target_final_pool_size": catalog["target_final_pool_size"],
        "planned_selected_before_top_up": planned_selected_before_top_up,
        "planned_top_up_size": catalog["planned_top_up_size"],
        "top_up_source": catalog["top_up_source"],
        "sources": list(source_index.values()),
        "source_index": source_index,
    }


def prepare_local_workspace(workspace_dir: str) -> Dict:
    plan = build_download_plan()
    workspace = Path(workspace_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    for source in plan["sources"]:
        source_dir = workspace / source["name"]
        source_dir.mkdir(exist_ok=True)

        for bucket in source.get("buckets", []):
            (source_dir / bucket["name"]).mkdir(exist_ok=True)

    return {
        "workspace_dir": str(workspace),
        "plan": plan,
    }


__all__ = [
    "build_download_plan",
    "prepare_local_workspace",
]
