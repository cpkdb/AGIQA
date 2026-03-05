#!/usr/bin/env python3
"""
Rule-based stratified sampler for the first public prompt working pool.

This stage consumes tagged prompt candidates and applies source-level quotas,
bucket-level quotas where configured, and a simple top-up pass from the
configured fallback source.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "prompt_source_catalog.json"
SAMPLING_SPEC_PATH = DATA_DIR / "prompt_sampling_spec.json"

SPATIAL_WORDS = {
    "left", "right", "above", "below", "behind", "in front of", "between",
    "next to", "beside", "near", "under", "over",
}
COUNT_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "single", "pair", "double", "triple",
}
ACTION_WORDS = {
    "holding", "running", "walking", "dancing", "jumping", "sitting", "standing",
    "smiling", "laughing", "hugging", "pointing", "looking", "posing",
}
EXPRESSION_WORDS = {
    "smiling", "laughing", "crying", "angry", "sad", "happy", "expression",
    "grinning", "frowning",
}
TEXT_WORDS = {
    "sign", "signage", "poster", "label", "logo", "book", "title", "text",
    "words", "letters", "saying",
}
REFLECTION_WORDS = {
    "mirror", "reflection", "reflective", "water", "lake", "pond", "glass", "window",
}
INTERACTION_WORDS = {
    "hugging", "talking", "fighting", "kissing", "holding hands", "facing each other",
    "together", "group", "crowd",
}
CREATIVE_WORDS = {
    "surreal", "fantasy", "dreamlike", "floating", "magical", "cinematic", "epic",
}


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_plan() -> Dict:
    catalog = _read_json(CATALOG_PATH)
    sampling_spec = _read_json(SAMPLING_SPEC_PATH)
    bucket_index = {item["name"]: item["buckets"] for item in sampling_spec["sources"]}
    sources = []
    for source in catalog["sources"]:
        item = dict(source)
        if item["name"] in bucket_index:
            item["buckets"] = bucket_index[item["name"]]
        sources.append(item)
    return {
        "target_final_pool_size": catalog["target_final_pool_size"],
        "planned_top_up_size": catalog["planned_top_up_size"],
        "top_up_source": catalog["top_up_source"],
        "sources": sources,
    }


def _has_tag(record: Dict, tag_name: str) -> bool:
    return tag_name in set(record.get("semantic_tags", []))


def _contains_any(prompt: str, phrases: Iterable[str]) -> bool:
    text = (prompt or "").lower()
    return any(phrase in text for phrase in phrases)


def assign_bucket(record: Dict, source_name: str) -> Optional[str]:
    """Assign a record to the first matching bucket for its source."""
    prompt = record.get("prompt", "")
    has_person = _has_tag(record, "has_person")
    has_face = _has_tag(record, "has_face")
    has_hand = _has_tag(record, "has_hand")
    has_animal = _has_tag(record, "has_animal")
    has_multiple = _has_tag(record, "has_multiple_objects")
    has_countable = _has_tag(record, "has_countable_objects")
    has_text = _has_tag(record, "has_text") or _has_tag(record, "has_logo_or_symbol")
    has_reflection = _has_tag(record, "has_reflective_surface")

    if source_name == "diffusiondb":
        if has_person and (has_face or has_hand or _contains_any(prompt, ACTION_WORDS | {"portrait"})):
            return "people_face_hands"
        if has_animal:
            return "animals"
        if has_multiple and _contains_any(prompt, SPATIAL_WORDS):
            return "multi_object_spatial"
        if has_countable and _contains_any(prompt, COUNT_WORDS):
            return "count_constraints"
        if has_text and _contains_any(prompt, TEXT_WORDS):
            return "text_sign_logo"
        if has_reflection and _contains_any(prompt, REFLECTION_WORDS):
            return "reflection_optics"
        return "general_scenes"

    if source_name == "pick_a_pic_v2":
        if has_person and _contains_any(prompt, ACTION_WORDS):
            return "people_and_activities"
        if (has_multiple or (has_person and _contains_any(prompt, INTERACTION_WORDS))):
            return "multi_subject_interaction"
        if _contains_any(prompt, ACTION_WORDS) and _contains_any(prompt, EXPRESSION_WORDS):
            return "explicit_action_expression"
        return "creative_but_structured"

    return None


def _shuffle_copy(items: List[Dict], seed: int) -> List[Dict]:
    copied = list(items)
    rng = random.Random(seed)
    rng.shuffle(copied)
    return copied


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


def sample_working_pool(
    records: Iterable[Dict],
    plan: Optional[Dict] = None,
    seed: int = 20260304,
) -> Tuple[List[Dict], Dict]:
    """Sample a first-pass 30k working pool (or a test plan-sized subset)."""
    if plan is None:
        plan = _default_plan()

    by_source: Dict[str, List[Dict]] = {}
    for record in records:
        by_source.setdefault(record.get("source", "unknown"), []).append(dict(record))

    sampled: List[Dict] = []
    used_keys = set()
    source_counts: Dict[str, int] = {}
    bucket_counts: Dict[str, Dict[str, int]] = {}

    for source_idx, source_plan in enumerate(plan["sources"]):
        source_name = source_plan["name"]
        source_records = _shuffle_copy(by_source.get(source_name, []), seed + source_idx)
        selected_for_source: List[Dict] = []

        if source_plan.get("buckets"):
            bucket_counts[source_name] = {}
            remaining: List[Dict] = []
            bucketed: Dict[str, List[Dict]] = {}

            for record in source_records:
                key = (record.get("source"), record.get("record_id"), record.get("prompt"))
                if key in used_keys:
                    continue
                bucket = assign_bucket(record, source_name)
                if bucket is None:
                    remaining.append(record)
                    continue
                bucketed.setdefault(bucket, []).append(record)

            for bucket in source_plan["buckets"]:
                bucket_name = bucket["name"]
                target = bucket["final_selected_size"]
                candidates = bucketed.get(bucket_name, [])
                chosen = candidates[:target]
                bucket_counts[source_name][bucket_name] = len(chosen)
                for record in chosen:
                    key = (record.get("source"), record.get("record_id"), record.get("prompt"))
                    if key not in used_keys:
                        used_keys.add(key)
                        record["assigned_bucket"] = bucket_name
                        selected_for_source.append(record)

            # Fill unmet source quota from remaining unselected source records.
            selected_keys = {
                (item.get("source"), item.get("record_id"), item.get("prompt"))
                for item in selected_for_source
            }
            spillover = []
            for record in source_records:
                key = (record.get("source"), record.get("record_id"), record.get("prompt"))
                if key not in selected_keys and key not in used_keys:
                    spillover.append(record)

            deficit = max(0, source_plan["final_selected_size"] - len(selected_for_source))
            for record in spillover[:deficit]:
                key = (record.get("source"), record.get("record_id"), record.get("prompt"))
                used_keys.add(key)
                record["assigned_bucket"] = "source_fill"
                selected_for_source.append(record)
        else:
            take = source_plan["final_selected_size"]
            for record in source_records[:take]:
                key = (record.get("source"), record.get("record_id"), record.get("prompt"))
                if key in used_keys:
                    continue
                used_keys.add(key)
                selected_for_source.append(record)

        sampled.extend(selected_for_source)
        source_counts[source_name] = len(selected_for_source)

    top_up_target = plan.get("planned_top_up_size", 0)
    top_up_source = plan.get("top_up_source")
    top_up_selected = 0
    if top_up_target and top_up_source:
        top_up_records = _shuffle_copy(by_source.get(top_up_source, []), seed + 999)
        for record in top_up_records:
            if top_up_selected >= top_up_target:
                break
            key = (record.get("source"), record.get("record_id"), record.get("prompt"))
            if key in used_keys:
                continue
            used_keys.add(key)
            record = dict(record)
            record["assigned_bucket"] = "top_up"
            sampled.append(record)
            source_counts[top_up_source] = source_counts.get(top_up_source, 0) + 1
            top_up_selected += 1

    sampled = sampled[: plan["target_final_pool_size"]]

    summary = {
        "output_count": len(sampled),
        "target_final_pool_size": plan["target_final_pool_size"],
        "source_counts": source_counts,
        "bucket_counts": bucket_counts,
        "top_up": {
            "source": top_up_source,
            "target_count": top_up_target,
            "selected_count": top_up_selected,
        },
    }
    return sampled, summary


def sample_candidate_file(
    input_path: str,
    output_dir: str,
    plan: Optional[Dict] = None,
    seed: int = 20260304,
) -> Dict[str, str]:
    """Read tagged candidates, sample the working pool, and write outputs."""
    input_file = Path(input_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(input_file)
    sampled, summary = sample_working_pool(records, plan=plan, seed=seed)

    sampled_path = output_root / "working_pool_v1.jsonl"
    summary_path = output_root / "sampling_summary.json"

    _write_jsonl(sampled_path, sampled)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "sampled_path": str(sampled_path),
        "summary_path": str(summary_path),
    }


__all__ = [
    "assign_bucket",
    "sample_candidate_file",
    "sample_working_pool",
]
