#!/usr/bin/env python3
"""
Clean and exact-deduplicate merged public prompt candidates.

This stage intentionally does only:
- prompt normalization
- empty / too-short filtering
- exact deduplication on normalized prompt text
- summary reporting

Near-duplicate detection and stratified sampling happen later.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_MIN_WORDS = 3
SOURCE_MIN_WORDS = {
    "t2i_compbench": 2,
}


def normalize_prompt(prompt: str) -> str:
    """Normalize whitespace and trim the prompt text."""
    if prompt is None:
        return ""
    return re.sub(r"\s+", " ", str(prompt)).strip()


def clean_and_dedup_candidates(
    records: Iterable[Dict],
    min_words: int = DEFAULT_MIN_WORDS,
) -> Tuple[List[Dict], Dict]:
    """Clean prompt records and remove exact duplicates."""
    cleaned: List[Dict] = []
    seen_prompts = set()

    input_count = 0
    filtered_empty_count = 0
    filtered_short_count = 0
    exact_duplicate_count = 0
    source_counts: Dict[str, int] = {}

    for record in records:
        input_count += 1
        normalized = normalize_prompt(record.get("prompt", ""))
        if not normalized:
            filtered_empty_count += 1
            continue

        source = record.get("source", "unknown")
        source_min_words = SOURCE_MIN_WORDS.get(source, min_words)
        if len(normalized.split()) < source_min_words:
            filtered_short_count += 1
            continue

        if normalized in seen_prompts:
            exact_duplicate_count += 1
            continue

        seen_prompts.add(normalized)

        item = dict(record)
        item["prompt"] = normalized
        cleaned.append(item)

        source_counts[source] = source_counts.get(source, 0) + 1

    summary = {
        "input_count": input_count,
        "filtered_empty_count": filtered_empty_count,
        "filtered_short_count": filtered_short_count,
        "exact_duplicate_count": exact_duplicate_count,
        "output_count": len(cleaned),
        "source_counts": source_counts,
        "min_words": min_words,
    }
    return cleaned, summary


def _read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_candidate_file(
    input_path: str,
    output_dir: str,
    min_words: int = DEFAULT_MIN_WORDS,
) -> Dict[str, str]:
    """Clean a merged candidate file and write cleaned outputs."""
    input_file = Path(input_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(input_file)
    cleaned, summary = clean_and_dedup_candidates(records, min_words=min_words)

    cleaned_path = output_root / "cleaned_exact_dedup.jsonl"
    summary_path = output_root / "cleaning_summary.json"

    _write_jsonl(cleaned_path, cleaned)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "cleaned_path": str(cleaned_path),
        "summary_path": str(summary_path),
    }


__all__ = [
    "clean_and_dedup_candidates",
    "clean_candidate_file",
    "normalize_prompt",
]
