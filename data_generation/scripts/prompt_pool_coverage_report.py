#!/usr/bin/env python3
"""
Generate a dimension coverage report for a sampled prompt working pool.

This stage evaluates how many prompts in the current working pool are compatible
with each degradation dimension under the existing Stage 1 filtering rules.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "config" / "semantic_tag_requirements.json"


def _load_prompt_filter_module():
    module_path = SCRIPT_DIR / "prompt_filter.py"
    spec = spec_from_file_location("prompt_filter", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _gap_level(available: int) -> str:
    if available < 100:
        return "scarce"
    if available < 500:
        return "limited"
    if available < 2000:
        return "healthy"
    return "strong"


def build_coverage_report(
    records: Iterable[Dict],
    tag_requirements: Optional[Dict[str, Dict]] = None,
    dimensions: Optional[List[str]] = None,
) -> Dict:
    """Build a dimension-level compatibility coverage report."""
    prompt_filter = _load_prompt_filter_module()
    if tag_requirements is None:
        tag_requirements = prompt_filter.load_tag_requirements(CONFIG_PATH)

    if dimensions is None:
        dimensions = list(tag_requirements.keys())

    record_list = list(records)
    total_prompts = len(record_list)
    dimension_stats: Dict[str, Dict] = {}

    for dimension in dimensions:
        available = 0
        for item in record_list:
            prompt = item.get("prompt", "")
            tags = item.get("semantic_tags", [])
            compatible, _ = prompt_filter.is_prompt_compatible(
                prompt=prompt,
                tags=tags,
                dimension=dimension,
                tag_requirements=tag_requirements,
            )
            if compatible:
                available += 1

        coverage = round((available / total_prompts) * 100, 2) if total_prompts else 0.0
        dimension_stats[dimension] = {
            "available": available,
            "total": total_prompts,
            "coverage": coverage,
            "gap_level": _gap_level(available),
        }

    scarcity_rank = sorted(
        dimension_stats.items(),
        key=lambda item: (item[1]["available"], item[0]),
    )

    return {
        "total_prompts": total_prompts,
        "dimensions": dimension_stats,
        "scarcity_ranking": [name for name, _ in scarcity_rank],
    }


def write_coverage_report(
    input_path: str,
    output_dir: str,
    dimensions: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Read a sampled working pool JSONL and write a coverage report JSON."""
    input_file = Path(input_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(input_file)
    report = build_coverage_report(records, dimensions=dimensions)

    report_path = output_root / "dimension_coverage_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"report_path": str(report_path)}


__all__ = [
    "build_coverage_report",
    "write_coverage_report",
]
