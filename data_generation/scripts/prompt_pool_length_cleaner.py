#!/usr/bin/env python3
"""Clean a merged prompt pool by prompt length and prompt-noise heuristics."""

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


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def clean_records(records: Iterable[Dict], **kwargs) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    cleaned: List[Dict] = []
    dropped: List[Dict] = []
    summary = {"kept": 0, "short": 0, "long": 0, "noise": 0}

    for record in records:
        item = dict(record)
        prompt = item.get("prompt", "")
        keep, reason = classify_prompt(prompt, **kwargs)
        if keep:
            cleaned.append(item)
            summary["kept"] += 1
        else:
            item["_drop_reason"] = reason
            dropped.append(item)
            summary[reason] += 1
    return cleaned, dropped, summary


def clean_prompt_pool(
    *,
    input_path: str,
    output_dir: str,
    min_words: int = DEFAULT_MIN_WORDS,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_words: int = DEFAULT_MAX_WORDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_commas: int = DEFAULT_MAX_COMMAS,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
) -> Dict[str, str]:
    input_file = Path(input_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(input_file)
    cleaned, dropped, summary = clean_records(
        records,
        min_words=min_words,
        min_chars=min_chars,
        max_words=max_words,
        max_chars=max_chars,
        max_commas=max_commas,
        max_repeat_ratio=max_repeat_ratio,
    )

    cleaned_path = output_root / "merged_working_pool_cleaned_v1.jsonl"
    dropped_path = output_root / "dropped_prompts_v1.jsonl"
    summary_path = output_root / "cleaning_summary_v1.json"

    _write_jsonl(cleaned_path, cleaned)
    _write_jsonl(dropped_path, dropped)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "cleaned_path": str(cleaned_path),
        "dropped_path": str(dropped_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a merged prompt pool by length and prompt-noise heuristics.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--min_chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max_words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--max_chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max_commas", type=int, default=DEFAULT_MAX_COMMAS)
    parser.add_argument("--max_repeat_ratio", type=float, default=DEFAULT_MAX_REPEAT_RATIO)
    args = parser.parse_args()

    outputs = clean_prompt_pool(
        input_path=args.input_path,
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
