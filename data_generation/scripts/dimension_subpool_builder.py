#!/usr/bin/env python3
"""
Build rule-based dimension-compatible prompt subpools from the current working pool.

This compiles the current prompt registry into `dimension -> compatible prompts`
artifacts so the generation agent can sample by target degradation dimension
without re-running compatibility checks every time.
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


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dimension_subpools(
    records: Iterable[Dict],
    dimensions: Optional[List[str]] = None,
) -> Dict:
    """Filter the working pool into dimension-specific compatible prompt lists."""
    prompt_filter = _load_prompt_filter_module()
    tag_requirements = prompt_filter.load_tag_requirements(CONFIG_PATH)
    if dimensions is None:
        dimensions = list(tag_requirements.keys())

    record_list = list(records)
    subpools: Dict[str, List[Dict]] = {}
    index: Dict[str, Dict] = {}

    for dimension in dimensions:
        compatible_records: List[Dict] = []
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
                compatible_records.append(dict(item))

        subpools[dimension] = compatible_records
        index[dimension] = {
            "count": len(compatible_records),
            "filename": f"{dimension}.jsonl",
        }

    return {
        "subpools": subpools,
        "index": index,
    }


def write_dimension_subpools(
    input_path: str,
    output_dir: str,
    dimensions: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Read a working pool JSONL and write per-dimension compatible subpools."""
    input_file = Path(input_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    subpool_dir = output_root / "dimension_subpools"
    subpool_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(input_file)
    result = build_dimension_subpools(records, dimensions=dimensions)

    for dimension, compatible_records in result["subpools"].items():
        _write_jsonl(subpool_dir / f"{dimension}.jsonl", compatible_records)

    index_payload = {
        "total_prompts": len(records),
        "dimensions": result["index"],
    }
    index_path = subpool_dir / "index.json"
    index_path.write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "subpool_dir": str(subpool_dir),
        "index_path": str(index_path),
    }


__all__ = [
    "build_dimension_subpools",
    "write_dimension_subpools",
]
