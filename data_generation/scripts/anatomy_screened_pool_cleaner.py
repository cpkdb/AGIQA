#!/usr/bin/env python3
"""Clean anatomy screened subpools by length and prompt-noise heuristics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_MIN_WORDS = 5
DEFAULT_MIN_CHARS = 25
DEFAULT_MAX_WORDS = 120
DEFAULT_MAX_CHARS = 800
DEFAULT_MAX_COMMAS = 20
DEFAULT_MAX_REPEAT_RATIO = 0.35


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _repeat_ratio(prompt: str) -> float:
    words = [w for w in re.split(r"\s+", prompt.strip()) if w]
    tokens = [re.sub(r"[^a-z0-9]+", "", word.lower()) for word in words]
    tokens = [token for token in tokens if token]
    if not tokens:
        return 0.0
    counter = Counter(tokens)
    repeated = sum(count - 1 for count in counter.values() if count > 1)
    return repeated / len(tokens)


def classify_prompt(
    prompt: str,
    *,
    min_words: int = DEFAULT_MIN_WORDS,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_words: int = DEFAULT_MAX_WORDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_commas: int = DEFAULT_MAX_COMMAS,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
) -> Tuple[bool, str]:
    words = [w for w in re.split(r"\s+", prompt.strip()) if w]
    char_len = len(prompt)
    if len(words) < min_words or char_len < min_chars:
        return False, "short"
    if len(words) > max_words or char_len > max_chars:
        return False, "long"
    repeat_ratio = _repeat_ratio(prompt)
    lowered = prompt.lower()
    if prompt.count(",") >= max_commas or repeat_ratio >= max_repeat_ratio or "http://" in lowered or "https://" in lowered:
        return False, "noise"
    return True, "keep"


def clean_records(records: Iterable[Dict], **kwargs) -> Tuple[List[Dict], Dict[str, int]]:
    cleaned: List[Dict] = []
    summary = {"kept": 0, "short": 0, "long": 0, "noise": 0}
    for record in records:
        prompt = record.get("prompt", "")
        keep, reason = classify_prompt(prompt, **kwargs)
        if keep:
            cleaned.append(dict(record))
            summary["kept"] += 1
        else:
            summary[reason] += 1
    return cleaned, summary


def clean_anatomy_screened_subpools(
    *,
    input_index_path: str,
    output_dir: str,
    min_words: int = DEFAULT_MIN_WORDS,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_words: int = DEFAULT_MAX_WORDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_commas: int = DEFAULT_MAX_COMMAS,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
) -> Dict[str, str]:
    input_index = Path(input_index_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_index.read_text(encoding="utf-8"))
    dimensions = dict(payload.get("dimensions", {}))

    cleaned_dimensions: Dict[str, Dict] = {}
    cleaning_summary: Dict[str, Dict[str, int]] = {}

    for dimension, meta in dimensions.items():
        filename = meta.get("filename")
        if not filename:
            continue

        source_path = Path(filename)
        if not source_path.is_absolute():
            source_path = input_index.parent / source_path

        if meta.get("skip_runtime_compat_filter"):
            records = _read_jsonl(source_path)
            cleaned, summary = clean_records(
                records,
                min_words=min_words,
                min_chars=min_chars,
                max_words=max_words,
                max_chars=max_chars,
                max_commas=max_commas,
                max_repeat_ratio=max_repeat_ratio,
            )
            out_path = output_root / f"{dimension}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for record in cleaned:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            cleaned_meta = dict(meta)
            cleaned_meta["count"] = len(cleaned)
            cleaned_meta["filename"] = str(out_path.resolve())
            cleaned_dimensions[dimension] = cleaned_meta
            cleaning_summary[dimension] = summary
        else:
            cleaned_dimensions[dimension] = dict(meta)

    cleaned_index = dict(payload)
    cleaned_index["dimensions"] = cleaned_dimensions
    cleaned_index["cleaning_rules"] = {
        "min_words": min_words,
        "min_chars": min_chars,
        "max_words": max_words,
        "max_chars": max_chars,
        "max_commas": max_commas,
        "max_repeat_ratio": max_repeat_ratio,
    }
    cleaned_index["cleaning_summary"] = cleaning_summary

    index_path = output_root / "index.json"
    summary_path = output_root / "cleaning_summary.json"
    index_path.write_text(json.dumps(cleaned_index, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(cleaning_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_root),
        "index_path": str(index_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean anatomy screened subpools by prompt length and noise heuristics.")
    parser.add_argument("--input_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--min_chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max_words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--max_chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max_commas", type=int, default=DEFAULT_MAX_COMMAS)
    parser.add_argument("--max_repeat_ratio", type=float, default=DEFAULT_MAX_REPEAT_RATIO)
    args = parser.parse_args()

    outputs = clean_anatomy_screened_subpools(
        input_index_path=args.input_index,
        output_dir=args.output_dir,
        min_words=args.min_words,
        min_chars=args.min_chars,
        max_words=args.max_words,
        max_chars=args.max_chars,
        max_commas=args.max_commas,
        max_repeat_ratio=args.max_repeat_ratio,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
