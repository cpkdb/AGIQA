#!/usr/bin/env python3
"""Build SD3.5 Turbo-specific short prompt pools using local CLIP token lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


DEFAULT_GLOBAL_MAX_TOKENS = 50
DEFAULT_STRICT_MAX_TOKENS = 40
STRICT_DIMENSIONS = {
    "awkward_framing",
    "awkward_positioning",
    "body_proportion_error",
    "cluttered_scene",
    "color_cast",
    "hand_malformation",
    "lighting_imbalance",
    "low_contrast",
    "overexposure",
    "underexposure",
    "unbalanced_layout",
}


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_clip_token_counter(tokenizer_path: str, tokenizer2_path: str) -> Callable[[str], int]:
    from transformers import CLIPTokenizer

    tok1 = CLIPTokenizer.from_pretrained(tokenizer_path)
    tok2 = CLIPTokenizer.from_pretrained(tokenizer2_path)

    def measure_tokens(prompt: str) -> int:
        ids1 = tok1(prompt, truncation=False)["input_ids"]
        ids2 = tok2(prompt, truncation=False)["input_ids"]
        return max(len(ids1), len(ids2))

    return measure_tokens


def filter_records_by_clip_tokens(
    records: Iterable[Dict],
    *,
    max_tokens: int,
    measure_tokens: Callable[[str], int],
) -> Tuple[List[Dict], Dict[str, int]]:
    cleaned: List[Dict] = []
    summary = {"kept": 0, "dropped_too_long": 0}

    for record in records:
        prompt = record.get("prompt", "")
        if measure_tokens(prompt) <= max_tokens:
            cleaned.append(dict(record))
            summary["kept"] += 1
        else:
            summary["dropped_too_long"] += 1

    return cleaned, summary


def build_sd35_turbo_pools(
    *,
    source_prompts_path: str,
    dimension_index_path: str,
    output_root: str,
    tokenizer_path: str,
    tokenizer2_path: str,
    global_max_tokens: int = DEFAULT_GLOBAL_MAX_TOKENS,
    strict_max_tokens: int = DEFAULT_STRICT_MAX_TOKENS,
) -> Dict[str, str]:
    source_path = Path(source_prompts_path).resolve()
    index_path = Path(dimension_index_path).resolve()
    output_dir = Path(output_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    measure_tokens = build_clip_token_counter(tokenizer_path, tokenizer2_path)

    global_records = _read_jsonl(source_path)
    turbo_records, global_summary = filter_records_by_clip_tokens(
        global_records,
        max_tokens=global_max_tokens,
        measure_tokens=measure_tokens,
    )

    turbo_pool_path = output_dir / "merged_working_pool_sd35_turbo_clipsafe_v1.jsonl"
    _write_jsonl(turbo_pool_path, turbo_records)

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    dimensions = dict(payload.get("dimensions", {}))

    subpool_dir = output_dir / "sd35_turbo_dimension_subpools_clipsafe_v1"
    subpool_dir.mkdir(parents=True, exist_ok=True)

    cleaned_dimensions: Dict[str, Dict] = {}
    cleaning_summary: Dict[str, Dict[str, int]] = {}

    for dimension, meta in dimensions.items():
        filename = meta.get("filename")
        if not filename:
            continue

        source_file = Path(filename)
        if not source_file.is_absolute():
            source_file = index_path.parent / source_file

        threshold = strict_max_tokens if dimension in STRICT_DIMENSIONS else global_max_tokens
        records = _read_jsonl(source_file)
        cleaned, summary = filter_records_by_clip_tokens(
            records,
            max_tokens=threshold,
            measure_tokens=measure_tokens,
        )

        out_path = subpool_dir / f"{dimension}.jsonl"
        _write_jsonl(out_path, cleaned)

        cleaned_meta = dict(meta)
        cleaned_meta["count"] = len(cleaned)
        cleaned_meta["filename"] = str(out_path.resolve())
        cleaned_dimensions[dimension] = cleaned_meta
        cleaning_summary[dimension] = {
            **summary,
            "max_tokens": threshold,
        }

    cleaned_index = dict(payload)
    cleaned_index["dimensions"] = cleaned_dimensions
    cleaned_index["cleaning_rules"] = {
        "global_max_tokens": global_max_tokens,
        "strict_max_tokens": strict_max_tokens,
        "strict_dimensions": sorted(STRICT_DIMENSIONS),
    }

    turbo_index_path = subpool_dir / "index.json"
    summary_path = output_dir / "sd35_turbo_pool_cleaning_summary.json"
    turbo_index_path.write_text(json.dumps(cleaned_index, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "global_pool": global_summary,
                "dimension_subpools": cleaning_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "turbo_pool_path": str(turbo_pool_path),
        "turbo_index_path": str(turbo_index_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SD3.5 Turbo short prompt pools from cleaned prompt data.")
    parser.add_argument("--source_prompts", required=True)
    parser.add_argument("--dimension_index", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--tokenizer2_path", required=True)
    parser.add_argument("--global_max_tokens", type=int, default=DEFAULT_GLOBAL_MAX_TOKENS)
    parser.add_argument("--strict_max_tokens", type=int, default=DEFAULT_STRICT_MAX_TOKENS)
    args = parser.parse_args()

    outputs = build_sd35_turbo_pools(
        source_prompts_path=args.source_prompts,
        dimension_index_path=args.dimension_index,
        output_root=args.output_root,
        tokenizer_path=args.tokenizer_path,
        tokenizer2_path=args.tokenizer2_path,
        global_max_tokens=args.global_max_tokens,
        strict_max_tokens=args.strict_max_tokens,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
