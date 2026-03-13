#!/usr/bin/env python3
"""Build screened anatomy dimension subpools from LLM screening results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SCREENED_LABELS = {"pass", "uncertain"}
BODY_CORE_DIMENSION = "body_proportion_error"
FACE_DELTA_DIMENSION = "face_asymmetry"
HAND_DELTA_DIMENSION = "hand_malformation"
ANIMAL_DIMENSION = "animal_anatomy_error"
HUMAN_CORE_SHARED_DIMENSIONS = (
    "body_proportion_error",
    "impossible_pose",
    "extra_limbs",
    "expression_mismatch",
)


def _is_expression_compatible(record: Dict) -> bool:
    signature = record.get("signature", {})
    return bool(signature.get("has_person")) and bool(signature.get("has_face"))


def _is_body_compatible(record: Dict) -> bool:
    signature = record.get("signature", {})
    return bool(signature.get("has_person")) and bool(signature.get("has_full_body"))


def _filter_records_for_dimension(records: Sequence[Dict], *, dimension: str) -> List[Dict]:
    if dimension == "expression_mismatch":
        return [record for record in records if _is_expression_compatible(record)]
    if dimension in {"body_proportion_error", "impossible_pose", "extra_limbs"}:
        return [record for record in records if _is_body_compatible(record)]
    return list(records)


def _read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _unique_prompts(rows: Iterable[Dict], *, dimension: str) -> List[str]:
    seen = set()
    prompts: List[str] = []
    for row in rows:
        if row.get("dimension") != dimension:
            continue
        if row.get("label") not in SCREENED_LABELS:
            continue
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts


def _index_source_records(source_records: Sequence[Dict]) -> Dict[str, Dict]:
    indexed: Dict[str, Dict] = {}
    for record in source_records:
        prompt = record.get("prompt")
        if isinstance(prompt, str) and prompt and prompt not in indexed:
            indexed[prompt] = record
    return indexed


def _records_from_prompts(prompt_lookup: Dict[str, Dict], prompts: Iterable[str]) -> List[Dict]:
    records: List[Dict] = []
    seen = set()
    for prompt in prompts:
        record = prompt_lookup.get(prompt)
        if record is None or prompt in seen:
            continue
        seen.add(prompt)
        records.append(dict(record))
    return records


def build_screened_dimension_subpools(
    *,
    source_records: Sequence[Dict],
    body_results: Sequence[Dict],
    hand_face_results: Sequence[Dict],
    animal_results: Sequence[Dict],
) -> Dict[str, Dict]:
    prompt_lookup = _index_source_records(source_records)

    body_core_prompts = _unique_prompts(body_results, dimension=BODY_CORE_DIMENSION)
    face_delta_prompts = _unique_prompts(hand_face_results, dimension=FACE_DELTA_DIMENSION)
    hand_delta_prompts = _unique_prompts(hand_face_results, dimension=HAND_DELTA_DIMENSION)
    animal_prompts = _unique_prompts(animal_results, dimension=ANIMAL_DIMENSION)

    body_core_records = _records_from_prompts(prompt_lookup, body_core_prompts)
    face_records = _records_from_prompts(prompt_lookup, [*body_core_prompts, *face_delta_prompts])
    hand_records = _records_from_prompts(prompt_lookup, [*body_core_prompts, *hand_delta_prompts])
    animal_records = _records_from_prompts(prompt_lookup, animal_prompts)

    subpools: Dict[str, List[Dict]] = {
        BODY_CORE_DIMENSION: _filter_records_for_dimension(body_core_records, dimension=BODY_CORE_DIMENSION),
        "impossible_pose": _filter_records_for_dimension(body_core_records, dimension="impossible_pose"),
        "extra_limbs": _filter_records_for_dimension(body_core_records, dimension="extra_limbs"),
        "expression_mismatch": _filter_records_for_dimension(body_core_records, dimension="expression_mismatch"),
        FACE_DELTA_DIMENSION: face_records,
        HAND_DELTA_DIMENSION: hand_records,
        ANIMAL_DIMENSION: animal_records,
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
        "stats": {
            "body_core_count": len(body_core_records),
            "face_delta_count": len(_records_from_prompts(prompt_lookup, face_delta_prompts)),
            "hand_delta_count": len(_records_from_prompts(prompt_lookup, hand_delta_prompts)),
            "animal_core_count": len(animal_records),
        },
    }


def write_screened_dimension_subpools(
    *,
    source_prompts_path: str,
    body_results_path: str,
    hand_face_results_path: str,
    animal_results_path: str,
    base_index_path: str,
    output_dir: str,
) -> Dict[str, str]:
    source_records = _read_jsonl(Path(source_prompts_path))
    body_results = _read_jsonl(Path(body_results_path))
    hand_face_results = _read_jsonl(Path(hand_face_results_path))
    animal_results = _read_jsonl(Path(animal_results_path))

    result = build_screened_dimension_subpools(
        source_records=source_records,
        body_results=body_results,
        hand_face_results=hand_face_results,
        animal_results=animal_results,
    )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for dimension, records in result["subpools"].items():
        out_path = out_dir / f"{dimension}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    base_index_file = Path(base_index_path).resolve()
    base_payload = json.loads(base_index_file.read_text(encoding="utf-8"))
    base_dimensions = dict(base_payload.get("dimensions", {}))
    base_subpool_dir = base_index_file.parent

    merged_dimensions: Dict[str, Dict] = {}
    for dimension, meta in base_dimensions.items():
        filename = meta.get("filename")
        if not filename:
            continue
        merged_dimensions[dimension] = dict(meta)
        merged_dimensions[dimension]["filename"] = str((base_subpool_dir / filename).resolve())

    for dimension, meta in result["index"].items():
        merged_dimensions[dimension] = dict(meta)
        merged_dimensions[dimension]["filename"] = str((out_dir / meta["filename"]).resolve())

    index_payload = {
        "schema_version": "1.0",
        "source_index": str(base_index_file),
        "dimensions": merged_dimensions,
        "notes": {
            "body_proportion_error": "Shared human core pool (pass + uncertain).",
            "impossible_pose": "Shared human core pool (pass + uncertain).",
            "extra_limbs": "Shared human core pool (pass + uncertain).",
            "expression_mismatch": "Shared human core pool (pass + uncertain).",
            "face_asymmetry": "Shared human core plus screened face delta.",
            "hand_malformation": "Shared human core plus screened hand delta.",
            "animal_anatomy_error": "Screened animal pool (pass + uncertain).",
        },
        "stats": result["stats"],
    }

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "index_path": str(index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build screened anatomy subpools from LLM screening results.")
    parser.add_argument("--source_prompts", required=True)
    parser.add_argument("--body_results", required=True)
    parser.add_argument("--hand_face_results", required=True)
    parser.add_argument("--animal_results", required=True)
    parser.add_argument("--base_index", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    outputs = write_screened_dimension_subpools(
        source_prompts_path=args.source_prompts,
        body_results_path=args.body_results,
        hand_face_results_path=args.hand_face_results,
        animal_results_path=args.animal_results,
        base_index_path=args.base_index,
        output_dir=args.output_dir,
    )

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
