#!/usr/bin/env python3
"""
Rule-based coarse semantic tagging for cleaned prompt candidates.

This stage reuses the existing SemanticRouter keyword system to attach
`semantic_tags` and `signature` to each prompt, then writes simple tag coverage
statistics for later sampling decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "config" / "semantic_tag_requirements.json"


def _load_tag_config() -> Dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _build_keyword_index(tag_config: Dict) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for tag_name, tag_info in tag_config.get("tags", {}).items():
        keywords = tag_info.get("detection_keywords", [])
        index[tag_name] = [keyword.lower() for keyword in keywords]
    return index


def _analyze_prompt(prompt: str, keyword_index: Dict[str, List[str]], tag_config: Dict) -> Dict:
    prompt_text = prompt or ""
    prompt_lower = prompt_text.lower()
    tags: List[str] = []

    for tag_name, keywords in keyword_index.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                tags.append(tag_name)
                break

    if re.search(r'[\'"][^\'"]{2,}[\'"]', prompt_text):
        tags.append("has_quoted_text")

    unique_tags = sorted(set(tags))
    signature = {"tags": unique_tags}
    for tag_name in tag_config.get("tags", {}):
        signature[tag_name] = tag_name in unique_tags
    signature["has_quoted_text"] = "has_quoted_text" in unique_tags
    return signature


def tag_candidates(records: Iterable[Dict]) -> Tuple[List[Dict], Dict]:
    """Tag prompt candidates using the existing keyword config."""
    tag_config = _load_tag_config()
    keyword_index = _build_keyword_index(tag_config)

    tagged: List[Dict] = []
    tag_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    input_count = 0

    for record in records:
        input_count += 1
        item = dict(record)
        signature = _analyze_prompt(item.get("prompt", ""), keyword_index, tag_config)
        semantic_tags = signature.get("tags", [])
        item["semantic_tags"] = semantic_tags
        item["signature"] = signature
        tagged.append(item)

        source = item.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

        for tag in semantic_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    summary = {
        "input_count": input_count,
        "output_count": len(tagged),
        "tag_counts": dict(sorted(tag_counts.items())),
        "source_counts": source_counts,
    }
    return tagged, summary


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


def tag_candidate_file(
    input_path: str,
    output_dir: str,
) -> Dict[str, str]:
    """Tag a cleaned candidate file and write tagged outputs."""
    input_file = Path(input_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(input_file)
    tagged, summary = tag_candidates(records)

    tagged_path = output_root / "tagged_candidates.jsonl"
    summary_path = output_root / "tagging_summary.json"

    _write_jsonl(tagged_path, tagged)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "tagged_path": str(tagged_path),
        "summary_path": str(summary_path),
    }


__all__ = [
    "tag_candidates",
    "tag_candidate_file",
]
