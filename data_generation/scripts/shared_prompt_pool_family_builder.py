#!/usr/bin/env python3
"""Build shared prompt-pool family candidate sets for later semantic screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prompt_pool_agent_tools as prompt_pool_tools


FULL_BODY_HINTS = (
    "full body",
    "full-body",
    "full length",
    "full-length",
    "head to toe",
    "head-to-toe",
    "standing",
    "walking",
    "running",
    "dancing",
    "posing",
)

NON_NATURAL_HUMAN_PRESENTATION_HINTS = (
    "armor",
    "armour",
    "armored",
    "armoured",
    "knight",
    "mecha",
    "cyborg",
    "android",
    "robotic",
    "exosuit",
    "power armor",
    "helmet",
    "full helmet",
    "spacesuit",
    "astronaut suit",
    "mascot",
    "elf",
    "vampire",
    "paladin",
    "barbarian",
    "super hero",
    "superhero",
    "demon",
    "fairy",
)

MULTI_PERSON_HINTS = (
    "group of",
    "crowd",
    "duo",
    "couple",
    "together",
    "friends",
    "two women",
    "two men",
    "two people",
    "three people",
)

BED_OR_ADULT_POSE_HINTS = (
    "in bed",
    "lying",
    "reclining",
    "sensual",
    "lingerie",
    "nude",
    "naked",
    "topless",
)

RELATION_HINTS = (
    "beside",
    "next to",
    "between",
    "near",
    "with",
    "alongside",
    "facing",
    "holding",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_tags(record: Dict[str, Any]) -> List[str]:
    tags = record.get("semantic_tags", record.get("tags", []))
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, list):
        return [tag for tag in tags if isinstance(tag, str)]
    return []


def _normalize_signature(record: Dict[str, Any]) -> Dict[str, Any]:
    signature = record.get("signature", record.get("prompt_signature"))
    if isinstance(signature, dict):
        return dict(signature)
    return {}


def _has_base_tags(record: Dict[str, Any], base_tags: Sequence[str]) -> bool:
    tags = set(_normalize_tags(record))
    signature = _normalize_signature(record)
    for tag in base_tags:
        if tag in tags:
            continue
        if bool(signature.get(tag)):
            continue
        return False
    return True


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = text.lower()
    normalized_phrase = phrase.lower().strip()
    if not normalized_phrase:
        return False
    pattern = r"\b" + re.escape(normalized_phrase).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, normalized_text) is not None


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _sample_id(family_name: str, prompt: str, rank: int) -> str:
    digest = hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()[:12]
    return f"{family_name}-{rank:05d}-{digest}"


def _dimensions_for_family(screening_spec: Dict[str, Any], family_name: str) -> List[str]:
    dimensions: List[str] = []
    for dimension, override in screening_spec.get("dimension_overrides", {}).items():
        if override.get("shared_pool_family") == family_name:
            dimensions.append(dimension)
    return sorted(dimensions)


def _evaluate_human_full_body_realistic(record: Dict[str, Any], *, base_tags: Sequence[str]) -> Tuple[bool, Dict[str, Any], str | None]:
    if not _has_base_tags(record, base_tags):
        return False, {"matched_base_tags": False}, "missing_base_tags"

    prompt = str(record.get("prompt", "")).strip()
    lowered = prompt.lower()
    signature = _normalize_signature(record)
    has_full_body_signal = bool(signature.get("has_full_body")) or "has_full_body" in _normalize_tags(record)
    keyword_body_readable = _contains_any(lowered, FULL_BODY_HINTS)

    heuristics = {
        "matched_base_tags": True,
        "has_full_body_signal": has_full_body_signal,
        "keyword_body_readable": keyword_body_readable,
        "prompt_length": len(prompt),
    }

    if _contains_any(lowered, NON_NATURAL_HUMAN_PRESENTATION_HINTS):
        return False, heuristics, "contains_non_natural_or_identity_obscuring_human_presentation"
    if _contains_any(lowered, MULTI_PERSON_HINTS):
        return False, heuristics, "contains_multi_person_signal"
    if _contains_any(lowered, BED_OR_ADULT_POSE_HINTS):
        return False, heuristics, "contains_bed_or_adult_pose_signal"
    if not (has_full_body_signal or keyword_body_readable):
        return False, heuristics, "insufficient_body_readability"
    return True, heuristics, None


def _evaluate_record(
    *,
    record: Dict[str, Any],
    family_name: str,
    family_spec: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any], str | None]:
    base_tags = list(family_spec.get("base_tags", []))
    if family_name == "human_full_body_realistic":
        return _evaluate_human_full_body_realistic(record, base_tags=base_tags)
    if family_name == "structured_object_primary":
        return _evaluate_structured_object_primary(record, base_tags=base_tags)
    if family_name == "multi_object_reference":
        return _evaluate_multi_object_reference(record, base_tags=base_tags)

    if not _has_base_tags(record, base_tags):
        return False, {"matched_base_tags": False}, "missing_base_tags"
    return True, {"matched_base_tags": True}, None


def _evaluate_structured_object_primary(
    record: Dict[str, Any], *, base_tags: Sequence[str]
) -> Tuple[bool, Dict[str, Any], str | None]:
    if not _has_base_tags(record, base_tags):
        return False, {"matched_base_tags": False}, "missing_base_tags"

    tags = set(_normalize_tags(record))
    signature = _normalize_signature(record)
    has_biological_signal = (
        "has_person" in tags
        or "has_animal" in tags
        or bool(signature.get("has_person"))
        or bool(signature.get("has_animal"))
    )
    heuristics = {
        "matched_base_tags": True,
        "has_biological_signal": has_biological_signal,
        "prompt_length": len(str(record.get("prompt", ""))),
    }
    if has_biological_signal:
        return False, heuristics, "contains_biological_primary_signal"
    return True, heuristics, None


def _evaluate_multi_object_reference(
    record: Dict[str, Any], *, base_tags: Sequence[str]
) -> Tuple[bool, Dict[str, Any], str | None]:
    tags = set(_normalize_tags(record))
    signature = _normalize_signature(record)
    prompt = str(record.get("prompt", "")).strip()
    has_any_entity_signal = (
        "has_multiple_objects" in tags
        or "has_person" in tags
        or "has_animal" in tags
        or "has_structured_object" in tags
        or "has_countable_objects" in tags
        or bool(signature.get("has_multiple_objects"))
        or bool(signature.get("has_person"))
        or bool(signature.get("has_animal"))
        or bool(signature.get("has_structured_object"))
        or bool(signature.get("has_countable_objects"))
    )
    has_relation_hint = _contains_any(prompt, RELATION_HINTS)
    has_multi_signal = "has_multiple_objects" in tags or bool(signature.get("has_multiple_objects"))
    heuristics = {
        "matched_base_tags": _has_base_tags(record, base_tags),
        "has_any_entity_signal": has_any_entity_signal,
        "has_multi_signal": has_multi_signal,
        "has_relation_hint": has_relation_hint,
        "prompt_length": len(prompt),
    }
    if not has_any_entity_signal:
        return False, heuristics, "missing_entity_signal"
    if not (has_multi_signal or has_relation_hint):
        return False, heuristics, "insufficient_relation_or_reference_signal"
    return True, heuristics, None


def build_shared_prompt_pool_family(
    *,
    source_records: Sequence[Dict[str, Any]],
    family_name: str,
    family_spec: Dict[str, Any],
    pool_variant: str,
) -> Dict[str, Any]:
    seen_prompts: set[str] = set()
    candidates: List[Dict[str, Any]] = []
    screening_input: List[Dict[str, Any]] = []
    excluded_reason_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    base_rule_recall_count = 0

    for record in source_records:
        prompt = str(record.get("prompt", "")).strip()
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        if _has_base_tags(record, family_spec.get("base_tags", [])):
            base_rule_recall_count += 1

        keep, heuristics, reason = _evaluate_record(
            record=record,
            family_name=family_name,
            family_spec=family_spec,
        )
        if not keep:
            if reason:
                excluded_reason_counts[reason] += 1
            continue

        if heuristics.get("has_full_body_signal"):
            signal_counts["has_full_body_signal"] += 1
        if heuristics.get("keyword_body_readable"):
            signal_counts["keyword_body_readable"] += 1

        enriched = dict(record)
        enriched["shared_pool_family"] = family_name
        enriched["candidate_stage"] = "rule_recall_filtered"
        enriched["pool_variant"] = pool_variant
        enriched["family_heuristics"] = heuristics
        candidates.append(enriched)

    for rank, record in enumerate(candidates, start=1):
        prompt = str(record.get("prompt", "")).strip()
        screening_input.append(
            {
                "family_name": family_name,
                "sample_id": _sample_id(family_name, prompt, rank),
                "prompt": prompt,
                "semantic_tags": _normalize_tags(record),
                "signature": _normalize_signature(record),
                "heuristics": dict(record.get("family_heuristics", {})),
            }
        )

    return {
        "family_name": family_name,
        "pool_variant": pool_variant,
        "builder_mode": family_spec.get("builder_mode", "rule_recall_then_llm_screen"),
        "base_tags": list(family_spec.get("base_tags", [])),
        "screening_goal": family_spec.get("screening_goal", ""),
        "llm_prompt_focus": list(family_spec.get("llm_prompt_focus", [])),
        "candidates": candidates,
        "screening_input": screening_input,
        "stats": {
            "total_source_records": len(source_records),
            "base_rule_recall_count": base_rule_recall_count,
            "candidate_count": len(candidates),
            "screening_input_count": len(screening_input),
            "excluded_reason_counts": dict(excluded_reason_counts),
            "signal_counts": dict(signal_counts),
        },
    }


def write_shared_prompt_pool_family(
    *,
    source_prompts_path: str,
    output_root: str,
    family_name: str,
    pool_variant: str,
    screening_spec_path: str = str(prompt_pool_tools.DEFAULT_SCREENING_SPEC_PATH),
    build_targets_path: str = str(prompt_pool_tools.DEFAULT_BUILD_TARGETS_PATH),
) -> Dict[str, str]:
    source_path = Path(source_prompts_path).resolve()
    source_records = _read_jsonl(source_path)

    screening_spec = prompt_pool_tools.load_prompt_pool_screening_spec(Path(screening_spec_path))
    build_targets = prompt_pool_tools.load_prompt_pool_build_targets(Path(build_targets_path))

    family_spec = screening_spec.get("shared_pool_families", {}).get(family_name)
    if not isinstance(family_spec, dict):
        raise ValueError(f"Unknown shared pool family: {family_name}")

    variant_targets = build_targets.get("shared_pool_family_outputs", {}).get(pool_variant)
    if not isinstance(variant_targets, dict):
        raise ValueError(f"Unknown pool variant: {pool_variant}")

    directory_name = str(variant_targets.get("directory_name", "")).strip()
    index_name = str(variant_targets.get("index_name", "index.json")).strip() or "index.json"
    candidate_suffix = str(variant_targets.get("candidate_suffix", "_candidates.jsonl")).strip() or "_candidates.jsonl"
    screening_input_suffix = (
        str(variant_targets.get("screening_input_suffix", "_screening_input.jsonl")).strip()
        or "_screening_input.jsonl"
    )

    out_dir = Path(output_root).resolve() / directory_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = build_shared_prompt_pool_family(
        source_records=source_records,
        family_name=family_name,
        family_spec=family_spec,
        pool_variant=pool_variant,
    )
    target_dimensions = _dimensions_for_family(screening_spec, family_name)

    candidate_path = out_dir / f"{family_name}{candidate_suffix}"
    screening_input_path = out_dir / f"{family_name}{screening_input_suffix}"
    index_path = out_dir / index_name

    _write_jsonl(candidate_path, result["candidates"])
    screening_input_rows: List[Dict[str, Any]] = []
    for row in result["screening_input"]:
        enriched_row = dict(row)
        enriched_row["pool_variant"] = pool_variant
        enriched_row["base_tags"] = result["base_tags"]
        enriched_row["screening_goal"] = result["screening_goal"]
        enriched_row["llm_prompt_focus"] = result["llm_prompt_focus"]
        enriched_row["target_dimensions"] = target_dimensions
        screening_input_rows.append(enriched_row)
    _write_jsonl(screening_input_path, screening_input_rows)

    index_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "schema_version": "1.0",
        "pool_variant": pool_variant,
        "source_prompts": str(source_path),
        "families": {
            family_name: {
                "count": result["stats"]["candidate_count"],
                "builder_mode": result["builder_mode"],
                "base_tags": result["base_tags"],
                "screening_goal": result["screening_goal"],
                "llm_prompt_focus": result["llm_prompt_focus"],
                "target_dimensions": target_dimensions,
                "candidate_filename": str(candidate_path.resolve()),
                "screening_input_filename": str(screening_input_path.resolve()),
                "excluded_reason_counts": result["stats"]["excluded_reason_counts"],
                "signal_counts": result["stats"]["signal_counts"],
                "base_rule_recall_count": result["stats"]["base_rule_recall_count"],
            }
        },
    }
    _write_json(index_path, index_payload)

    return {
        "output_dir": str(out_dir),
        "candidate_path": str(candidate_path),
        "screening_input_path": str(screening_input_path),
        "index_path": str(index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shared prompt pool family candidate set.")
    parser.add_argument("--source_prompts", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--family_name", required=True)
    parser.add_argument("--pool_variant", choices=["common", "turbo"], required=True)
    parser.add_argument(
        "--screening_spec_path",
        default=str(prompt_pool_tools.DEFAULT_SCREENING_SPEC_PATH),
    )
    parser.add_argument(
        "--build_targets_path",
        default=str(prompt_pool_tools.DEFAULT_BUILD_TARGETS_PATH),
    )
    args = parser.parse_args()

    outputs = write_shared_prompt_pool_family(
        source_prompts_path=args.source_prompts,
        output_root=args.output_root,
        family_name=args.family_name,
        pool_variant=args.pool_variant,
        screening_spec_path=args.screening_spec_path,
        build_targets_path=args.build_targets_path,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
