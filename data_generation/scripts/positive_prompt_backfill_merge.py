#!/usr/bin/env python3
"""
Merge generated positive-prompt backfill results into the working pool and rebuild
derived prompt-pool artifacts.
"""

from __future__ import annotations

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_DIR = SCRIPT_DIR.parent / "data" / "prompt_sources_workspace"
DEFAULT_BASE_WORKING_POOL = DEFAULT_WORKSPACE_DIR / "working_pool_v1.jsonl"


def _load_module(module_path: Path, module_name: str):
    spec = spec_from_file_location(module_name, module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_tagger_module():
    return _load_module(SCRIPT_DIR / "prompt_candidate_tagger.py", "prompt_candidate_tagger")


def _load_macro_module():
    return _load_module(SCRIPT_DIR / "prompt_macro_taxonomy_report.py", "prompt_macro_taxonomy_report")


def _load_coverage_module():
    return _load_module(SCRIPT_DIR / "prompt_pool_coverage_report.py", "prompt_pool_coverage_report")


def _load_subpool_module():
    return _load_module(SCRIPT_DIR / "dimension_subpool_builder.py", "dimension_subpool_builder")


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


def _collect_generated_files(generated_inputs: Sequence[str]) -> List[Path]:
    collected: List[Path] = []
    seen = set()
    for raw_path in generated_inputs:
        path = Path(raw_path).resolve()
        matches: List[Path] = []
        if path.is_file() and path.suffix == ".jsonl":
            matches = [path]
        elif path.is_dir():
            direct = path / "generated_positive_prompts.jsonl"
            if direct.exists():
                matches.append(direct)
            matches.extend(sorted(candidate.resolve() for candidate in path.rglob("generated_positive_prompts.jsonl")))

        for match in matches:
            key = str(match)
            if key not in seen:
                seen.add(key)
                collected.append(match)

    if not collected:
        raise FileNotFoundError("No generated_positive_prompts.jsonl files found in generated_inputs")

    return collected


def _normalize_generated_records(records: Iterable[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for record in records:
        prompt = (record.get("prompt") or "").strip()
        if not prompt:
            continue
        dimension = record.get("dimension", "unknown")
        normalized.append(
            {
                "prompt": prompt,
                "source": "llm_backfill",
                "assigned_bucket": f"llm_backfill::{dimension}",
                "backfill_dimension": dimension,
                "backfill_macro_bucket": record.get("macro_bucket"),
                "backfill_request_id": record.get("request_id"),
                "backfill_prompt_index": record.get("prompt_index_in_request"),
                "backfill_model": record.get("model"),
                "backfill_coverage_mode": record.get("coverage_mode"),
            }
        )
    return normalized


def _dedup_new_records(base_records: Sequence[Dict], new_records: Sequence[Dict]) -> List[Dict]:
    seen = {((item.get("prompt") or "").strip()) for item in base_records}
    deduped: List[Dict] = []
    for item in new_records:
        prompt = (item.get("prompt") or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        deduped.append(dict(item))
    return deduped


def merge_generated_backfill(
    base_working_pool_path: str,
    generated_inputs: Sequence[str],
    output_dir: str,
) -> Dict[str, str]:
    base_path = Path(base_working_pool_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_records = _read_jsonl(base_path)

    generated_files = _collect_generated_files(generated_inputs)
    generated_raw_records: List[Dict] = []
    for path in generated_files:
        generated_raw_records.extend(_read_jsonl(path))

    normalized_new_records = _normalize_generated_records(generated_raw_records)
    deduped_new_records = _dedup_new_records(base_records, normalized_new_records)

    tagger = _load_tagger_module()
    tagged_new_records, tag_summary = tagger.tag_candidates(deduped_new_records)

    merged_records = [dict(item) for item in base_records] + [dict(item) for item in tagged_new_records]
    merged_working_pool_path = output_root / "merged_working_pool.jsonl"
    _write_jsonl(merged_working_pool_path, merged_records)

    macro_module = _load_macro_module()
    macro_result = macro_module.write_macro_reports(
        input_path=str(merged_working_pool_path),
        output_dir=str(output_root),
    )

    coverage_module = _load_coverage_module()
    coverage_result = coverage_module.write_coverage_report(
        input_path=macro_result["tagged_path"],
        output_dir=str(output_root),
    )

    subpool_module = _load_subpool_module()
    subpool_result = subpool_module.write_dimension_subpools(
        input_path=macro_result["tagged_path"],
        output_dir=str(output_root),
    )

    merge_summary = {
        "schema_version": "1.0",
        "base_prompt_count": len(base_records),
        "generated_file_count": len(generated_files),
        "generated_prompt_count": len(generated_raw_records),
        "normalized_generated_prompt_count": len(normalized_new_records),
        "deduped_new_prompt_count": len(deduped_new_records),
        "merged_prompt_count": len(merged_records),
        "new_prompt_source_counts": tag_summary.get("source_counts", {}),
        "generated_files": [str(path) for path in generated_files],
    }
    merge_summary_path = output_root / "merge_summary.json"
    merge_summary_path.write_text(
        json.dumps(merge_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "merged_working_pool_path": str(merged_working_pool_path),
        "merge_summary_path": str(merge_summary_path),
        "macro_tagged_path": macro_result["tagged_path"],
        "macro_distribution_path": macro_result["distribution_path"],
        "macro_matrix_path": macro_result["matrix_path"],
        "coverage_report_path": coverage_result["report_path"],
        "subpool_dir": subpool_result["subpool_dir"],
        "subpool_index_path": subpool_result["index_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge generated positive-prompt backfill outputs into the working pool and rebuild artifacts."
    )
    parser.add_argument(
        "--base_working_pool",
        type=str,
        default=str(DEFAULT_BASE_WORKING_POOL),
        help="Path to the base working pool JSONL.",
    )
    parser.add_argument(
        "--generated_inputs",
        type=str,
        required=True,
        help="Comma-separated list of generated JSONL files or batch directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_WORKSPACE_DIR / "backfill_merged"),
    )
    args = parser.parse_args()

    generated_inputs = [item.strip() for item in args.generated_inputs.split(",") if item.strip()]
    result = merge_generated_backfill(
        base_working_pool_path=args.base_working_pool,
        generated_inputs=generated_inputs,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
