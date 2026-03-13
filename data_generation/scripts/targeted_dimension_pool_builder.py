#!/usr/bin/env python3
"""Build lightweight targeted subpools for active stable object dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filter_unique_by_prompt(records: Iterable[Dict]) -> List[Dict]:
    seen = set()
    result: List[Dict] = []
    for record in records:
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt or prompt in seen:
            continue
        seen.add(prompt)
        result.append(dict(record))
    return result


def build_targeted_dimension_subpools(
    *,
    source_records: Sequence[Dict],
    object_shape_records: Sequence[Dict],
) -> Dict[str, Dict]:
    del source_records
    object_shape_unique = _filter_unique_by_prompt(object_shape_records)
    material_mismatch_records = list(object_shape_unique)

    subpools = {
        "object_structure_error": list(object_shape_unique),
        "material_mismatch": _filter_unique_by_prompt(material_mismatch_records),
    }

    index: Dict[str, Dict] = {}
    for dimension, records in subpools.items():
        index[dimension] = {
            "count": len(records),
            "filename": f"{dimension}.jsonl",
            "skip_runtime_compat_filter": True,
        }

    return {
        "subpools": subpools,
        "index": index,
    }


def write_targeted_dimension_subpools(
    *,
    source_prompts_path: str,
    base_index_path: str,
    output_dir: str,
) -> Dict[str, str]:
    source_path = Path(source_prompts_path).resolve()
    base_index_file = Path(base_index_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_records = _read_jsonl(source_path)
    base_payload = json.loads(base_index_file.read_text(encoding="utf-8"))
    base_dimensions = dict(base_payload.get("dimensions", {}))

    object_shape_meta = base_dimensions.get("object_shape_error")
    if not object_shape_meta or not object_shape_meta.get("filename"):
        raise ValueError("base index missing object_shape_error subpool")

    object_shape_path = Path(object_shape_meta["filename"])
    if not object_shape_path.is_absolute():
        object_shape_path = base_index_file.parent / object_shape_path
    object_shape_records = _read_jsonl(object_shape_path)

    result = build_targeted_dimension_subpools(
        source_records=source_records,
        object_shape_records=object_shape_records,
    )

    for dimension, records in result["subpools"].items():
        out_path = output_root / f"{dimension}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    merged_dimensions: Dict[str, Dict] = {}
    for dimension, meta in base_dimensions.items():
        merged_dimensions[dimension] = dict(meta)

    for dimension, meta in result["index"].items():
        merged_dimensions[dimension] = dict(meta)
        merged_dimensions[dimension]["filename"] = str((output_root / meta["filename"]).resolve())

    index_payload = {
        **base_payload,
        "source_index": str(base_index_file),
        "dimensions": merged_dimensions,
        "notes": {
            **base_payload.get("notes", {}),
            "object_structure_error": "Lightweight targeted pool derived from object_shape_error records.",
            "material_mismatch": "Lightweight targeted pool derived from object_shape_error records.",
        },
    }

    index_path = output_root / "index.json"
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_root),
        "index_path": str(index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight targeted subpools for new stable dimensions.")
    parser.add_argument("--source_prompts", required=True)
    parser.add_argument("--base_index", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    outputs = write_targeted_dimension_subpools(
        source_prompts_path=args.source_prompts,
        base_index_path=args.base_index,
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
