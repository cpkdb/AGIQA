#!/usr/bin/env python3
"""Family-level LLM semantic screening for shared prompt pool families."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import positive_prompt_semantic_screening as screening_tools
import prompt_pool_agent_tools as prompt_pool_tools


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


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


def _has_any_prefilter_tags(record: Dict[str, Any], tags_any: Sequence[str]) -> bool:
    tags = set(_normalize_tags(record))
    signature = _normalize_signature(record)
    for tag in tags_any:
        if tag in tags or bool(signature.get(tag)):
            return True
    return False


RELATION_HINTS: Tuple[str, ...] = (
    "beside",
    "next to",
    "between",
    "near",
    "with",
    "alongside",
    "facing",
    "holding",
)


def _contains_relation_hint(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(hint in lowered for hint in RELATION_HINTS)


def _passes_family_prefilter(record: Dict[str, Any], family_name: str, family_spec: Dict[str, Any]) -> bool:
    if family_name != "multi_object_reference":
        return _has_base_tags(record, list(family_spec.get("base_tags", [])))

    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        return False
    tags_any = list(family_spec.get("prefilter_tags_any", []))
    has_any_entity_signal = _has_any_prefilter_tags(record, tags_any)
    tags = set(_normalize_tags(record))
    signature = _normalize_signature(record)
    has_multi_signal = "has_multiple_objects" in tags or bool(signature.get("has_multiple_objects"))
    has_relation_hint = _contains_relation_hint(prompt)
    return has_any_entity_signal and (has_multi_signal or has_relation_hint)


def _target_dimensions_for_family(screening_spec: Dict[str, Any], family_name: str) -> List[str]:
    dimensions: List[str] = []
    for dimension, override in screening_spec.get("dimension_overrides", {}).items():
        if override.get("shared_pool_family") == family_name:
            dimensions.append(dimension)
    return sorted(dimensions)


def _family_prompt_rules(family_name: str) -> Dict[str, Any]:
    if family_name == "human_full_body_realistic":
        return {
            "system": (
                "You are a strict data quality reviewer for positive text-to-image prompts. "
                "Your task is to judge whether each prompt is suitable for the shared prompt-pool family named "
                "human_full_body_realistic. This family is used as the parent pool for later degradations such as "
                "body_proportion_error and extra_limbs. Use binary labels only: pass or fail. "
                "A prompt should be labeled pass only if it is likely to generate at least one readable main human subject "
                "whose body, limbs, or head-to-body relation is clear enough for later anatomical degradation. "
                "multiple people are allowed as long as one main human subject is still clearly readable. "
                "Do not reject a prompt only because it is stylized or cinematic if the main human body is still readable. "
                "Label fail if the prompt is likely to produce only face-closeups without body cues, a non-human or heavily "
                "transformed humanoid presentation, heavily obscured or structurally unreadable bodies, mannequins/statues/toys "
                "instead of real humans, or explicit bed/erotic/adult-pose content. Return JSON only."
            ),
            "pass_criteria": [
                "at least one readable main human subject",
                "body, limbs, or head-to-body relation is likely readable",
                "multi-person scenes are allowed if one main subject is still clearly readable",
                "stylization is allowed if the human body remains readable enough for later anatomy degradation",
            ],
            "fail_criteria": [
                "only face / portrait / close-up without usable body proportion cues",
                "non-human or heavily transformed humanoid presentation",
                "heavily obscured or structurally unreadable body",
                "mannequin / statue / toy / doll instead of a real human subject",
                "explicit bed / erotic / adult-pose content",
            ],
        }
    if family_name == "structured_object_primary":
        return {
            "system": (
                "You are a strict data quality reviewer for positive text-to-image prompts. "
                "Your task is to judge whether each prompt is suitable for the shared prompt-pool family named "
                "structured_object_primary. This family is used as the parent pool for later degradations such as "
                "object_structure_error and material_mismatch. Use binary labels only: pass or fail. "
                "A prompt should be labeled pass only if it is likely to generate one dominant structured non-living object "
                "whose shape, parts, or surface material remain easy to perceive. "
                "Do not reject a prompt only because it is stylized if the object itself remains clear and dominant. "
                "Label fail if the prompt is likely to make humans or animals the main subject, make the target object too minor "
                "or too crowded to read clearly, or describe mostly abstract scenery without a dominant structured object. "
                "Return JSON only."
            ),
            "pass_criteria": [
                "one dominant structured non-living object",
                "shape, parts, or surface material are likely readable",
                "stylization is allowed if the main object remains clear and dominant",
            ],
            "fail_criteria": [
                "humans or animals as the main subject",
                "the main object is too minor, too crowded, or too unclear to read well",
                "mostly abstract scenery or environment without a dominant structured object",
            ],
        }
    if family_name == "multi_object_reference":
        return {
            "system": (
                "You are a strict data quality reviewer for positive text-to-image prompts. "
                "Your task is to judge whether each prompt is suitable for the shared prompt-pool family named "
                "multi_object_reference. This family is used as the parent pool for later degradations such as "
                "scale_inconsistency and penetration_overlap. Use binary labels only: pass or fail. "
                "A prompt should be labeled pass only if it is likely to generate at least two readable entities, "
                "or one main subject with a clear reference context, so that size comparison or contact relations can be judged later. "
                "Entities may be objects, humans, or animals. "
                "Do not reject a prompt only because it is stylized if boundaries and relations remain readable. "
                "Label fail if the prompt is likely to produce only a single isolated subject without reference, "
                "or a scene where object boundaries and relations are too vague to compare. Return JSON only."
            ),
            "pass_criteria": [
                "at least two readable entities, or one main subject with a clear reference context",
                "entities may be objects, humans, or animals",
                "boundaries, contact, or size-reference relations are likely readable",
                "stylization is allowed if relations remain readable",
            ],
            "fail_criteria": [
                "single isolated subject without a useful reference context",
                "boundaries or object relations are too vague to compare",
                "scene is too abstract or too unreadable for later relation-based degradation",
            ],
        }
    return {
        "system": (
            "You are a strict data quality reviewer for positive text-to-image prompts. "
            "Judge whether each prompt is suitable for the requested shared prompt-pool family. "
            "Use binary labels only: pass or fail. Return JSON only."
        ),
        "pass_criteria": [
            "prompt likely contains the core semantic prerequisites for this shared family"
        ],
        "fail_criteria": [
            "prompt likely lacks the core semantic prerequisites for this shared family"
        ],
    }


def prepare_family_candidates(
    *,
    source_records: Sequence[Dict[str, Any]],
    family_name: str,
    family_spec: Dict[str, Any],
    target_dimensions: Sequence[str],
) -> List[Dict[str, Any]]:
    base_tags = list(family_spec.get("base_tags", []))
    seen_prompts: set[str] = set()
    candidates: List[Dict[str, Any]] = []
    rank = 0
    for record in source_records:
        prompt = str(record.get("prompt", "")).strip()
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        if not _passes_family_prefilter(record, family_name, family_spec):
            continue
        rank += 1
        candidates.append(
            {
                "family_name": family_name,
                "sample_id": f"{family_name}-{rank:05d}-{_prompt_hash(prompt)}",
                "prompt": prompt,
                "semantic_tags": _normalize_tags(record),
                "signature": _normalize_signature(record),
                "base_tags": base_tags,
                "screening_goal": family_spec.get("screening_goal", ""),
                "llm_prompt_focus": list(family_spec.get("llm_prompt_focus", [])),
                "target_dimensions": list(target_dimensions),
            }
        )
    return candidates


def select_candidate_shard(
    candidates: Sequence[Dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if num_shards == 1:
        return list(candidates)
    return [row for idx, row in enumerate(candidates) if idx % num_shards == shard_index]


def build_family_system_prompt(family_name: str) -> str:
    return str(_family_prompt_rules(family_name)["system"])


def build_family_user_prompt(
    *,
    family_name: str,
    family_spec: Dict[str, Any],
    target_dimensions: Sequence[str],
    batch: Sequence[Dict[str, Any]],
) -> str:
    rules = _family_prompt_rules(family_name)
    target_dimensions_str = ", ".join(target_dimensions)
    screening_goal = family_spec.get("screening_goal", "")
    llm_focus = list(family_spec.get("llm_prompt_focus", []))
    focus_lines = [f"- {item}" for item in llm_focus]
    pass_lines = [f"- {item}" for item in rules["pass_criteria"]]
    fail_lines = [f"- {item}" for item in rules["fail_criteria"]]

    lines = [
        f"Shared family: {family_name}",
        f"Target dimensions: {target_dimensions_str}",
        "",
        f"Screening goal: {screening_goal}",
        "",
        "Pass criteria:",
        *pass_lines,
        "",
        "Fail criteria:",
        *fail_lines,
        "",
        "Family focus cues:",
        *focus_lines,
        "",
        "Output JSON schema:",
        f'{{"family_name":"{family_name}","results":[{{"sample_id":"...","label":"pass|fail","reason":"..."}}]}}',
        "",
        "Prompts:",
    ]
    for item in batch:
        lines.append(f'- sample_id="{item["sample_id"]}" prompt="{item["prompt"]}"')
    return "\n".join(lines)


def _strip_code_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def _recover_result_items_from_malformed_json(raw_text: str) -> List[Dict[str, Any]]:
    text = _strip_code_fences(raw_text)
    decoder = json.JSONDecoder()
    recovered: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for match in re.finditer(r'\{\s*"sample_id"\s*:', text):
        try:
            item, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        recovered.append(
            {
                "sample_id": sample_id,
                "label": str(item.get("label", "fail")).strip().lower(),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return recovered


def parse_family_results(raw_text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        parsed = screening_tools._extract_json_obj(raw_text)
        items = parsed.get("results", [])
        if isinstance(items, list):
            return items, None
        return [], None
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
    recovered = _recover_result_items_from_malformed_json(raw_text)
    return recovered, parse_error


def write_family_screening_inputs(
    *,
    source_prompts_path: str,
    output_dir: str,
    family_name: str,
    screening_spec_path: str = str(prompt_pool_tools.DEFAULT_SCREENING_SPEC_PATH),
) -> Dict[str, str]:
    source_path = Path(source_prompts_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_records = _read_jsonl(source_path)
    screening_spec = prompt_pool_tools.load_prompt_pool_screening_spec(Path(screening_spec_path))
    family_spec = screening_spec.get("shared_pool_families", {}).get(family_name)
    if not isinstance(family_spec, dict):
        raise ValueError(f"Unknown shared pool family: {family_name}")
    target_dimensions = _target_dimensions_for_family(screening_spec, family_name)
    candidates = prepare_family_candidates(
        source_records=source_records,
        family_name=family_name,
        family_spec=family_spec,
        target_dimensions=target_dimensions,
    )

    input_jsonl = out_dir / f"{family_name}_family_screening_input.jsonl"
    manifest_json = out_dir / f"{family_name}_family_screening_manifest.json"
    _append_jsonl(input_jsonl, candidates)
    _write_json(
        manifest_json,
        {
            "generated_at": datetime.utcnow().isoformat(),
            "family_name": family_name,
            "source_prompts_path": str(source_path),
            "candidate_count": len(candidates),
            "base_tags": list(family_spec.get("base_tags", [])),
            "screening_goal": family_spec.get("screening_goal", ""),
            "llm_prompt_focus": list(family_spec.get("llm_prompt_focus", [])),
            "target_dimensions": target_dimensions,
            "input_jsonl": str(input_jsonl),
        },
    )
    return {
        "input_jsonl": str(input_jsonl),
        "manifest_json": str(manifest_json),
    }


def run_family_screening(args) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_config_path = Path(args.llm_config).resolve()
    config = screening_tools._load_llm_config(llm_config_path)
    llm_cfg = config.get("llm", {})
    model = args.model_override or llm_cfg.get("model", "gpt-5-chat-latest")
    client = screening_tools._create_openai_client(config)

    screening_spec = prompt_pool_tools.load_prompt_pool_screening_spec(Path(args.screening_spec_path))
    family_spec = screening_spec.get("shared_pool_families", {}).get(args.family_name)
    if not isinstance(family_spec, dict):
        raise ValueError(f"Unknown shared pool family: {args.family_name}")
    target_dimensions = _target_dimensions_for_family(screening_spec, args.family_name)

    all_candidates = prepare_family_candidates(
        source_records=_read_jsonl(Path(args.source_prompts).resolve()),
        family_name=args.family_name,
        family_spec=family_spec,
        target_dimensions=target_dimensions,
    )
    candidates = select_candidate_shard(
        all_candidates,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    manifest_path = output_dir / "family_screening_manifest.json"
    results_jsonl = output_dir / "family_screening_results.jsonl"
    raw_jsonl = output_dir / "family_screening_raw_llm_responses.jsonl"
    summary_path = output_dir / "family_screening_summary.json"

    _write_json(
        manifest_path,
        {
            "created_at": datetime.utcnow().isoformat(),
            "family_name": args.family_name,
            "source_prompts": str(Path(args.source_prompts).resolve()),
            "total_candidate_count": len(all_candidates),
            "candidate_count": len(candidates),
            "batch_size": args.batch_size,
            "model": model,
            "llm_config_path": str(llm_config_path),
            "target_dimensions": target_dimensions,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
    )

    system_prompt = build_family_system_prompt(args.family_name)
    request_counter = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    label_counts = {"pass": 0, "fail": 0, "total": 0}

    for i in range(0, len(candidates), args.batch_size):
        batch = candidates[i : i + args.batch_size]
        if not batch:
            continue
        request_counter += 1
        user_prompt = build_family_user_prompt(
            family_name=args.family_name,
            family_spec=family_spec,
            target_dimensions=target_dimensions,
            batch=batch,
        )
        raw_text, usage = screening_tools._call_llm_with_retry(
            client=client,
            llm_cfg=llm_cfg,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        for key in usage_totals:
            value = usage.get(key)
            if isinstance(value, int):
                usage_totals[key] += value

        items, parse_error = parse_family_results(raw_text)

        by_id = {item["sample_id"]: item for item in batch}
        out_rows: List[Dict[str, Any]] = []
        for result in items:
            sample_id = str(result.get("sample_id", "")).strip()
            if sample_id not in by_id:
                continue
            label = str(result.get("label", "fail")).strip().lower()
            if label != "pass":
                label = "fail"
            reason = str(result.get("reason", "")).strip()
            sample = by_id[sample_id]
            out_rows.append(
                {
                    "family_name": args.family_name,
                    "sample_id": sample_id,
                    "prompt": sample["prompt"],
                    "label": label,
                    "reason": reason,
                    "request_id": request_counter,
                    "model": model,
                    "keep_by_policy": label == "pass",
                }
            )

        returned_ids = {row["sample_id"] for row in out_rows}
        for sample_id, sample in by_id.items():
            if sample_id in returned_ids:
                continue
            out_rows.append(
                {
                    "family_name": args.family_name,
                    "sample_id": sample_id,
                    "prompt": sample["prompt"],
                    "label": "fail",
                    "reason": "missing_in_model_output",
                    "request_id": request_counter,
                    "model": model,
                    "keep_by_policy": False,
                }
            )

        _append_jsonl(results_jsonl, out_rows)
        _append_jsonl(
            raw_jsonl,
            [
                {
                    "family_name": args.family_name,
                    "request_id": request_counter,
                    "batch_size": len(batch),
                    "raw_content": raw_text,
                    "usage": usage,
                    "parse_error": parse_error,
                }
            ],
        )

        for row in out_rows:
            label_counts["total"] += 1
            label_counts[row["label"]] += 1

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary = {
        "created_at": datetime.utcnow().isoformat(),
        "family_name": args.family_name,
        "total_candidate_count": len(all_candidates),
        "candidate_count": len(candidates),
        "request_count": request_counter,
        "usage_totals": usage_totals,
        "pass": label_counts["pass"],
        "fail": label_counts["fail"],
        "pass_rate": round(label_counts["pass"] / label_counts["total"], 4) if label_counts["total"] else 0.0,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    _write_json(summary_path, summary)
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "results_jsonl": str(results_jsonl),
        "raw_jsonl": str(raw_jsonl),
        "summary_path": str(summary_path),
        "request_count": request_counter,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Family-level semantic screening for shared prompt pool families.")
    parser.add_argument("--llm_config", type=str, default=str(SCRIPT_DIR.parent / "config" / "llm_config_api_gpt_ge.yaml"))
    parser.add_argument("--source_prompts", type=str, required=True)
    parser.add_argument("--family_name", type=str, required=True)
    parser.add_argument("--screening_spec_path", type=str, default=str(prompt_pool_tools.DEFAULT_SCREENING_SPEC_PATH))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--sleep_seconds", type=float, default=0.3)
    parser.add_argument("--model_override", type=str, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()

    if args.prepare_only:
        result = write_family_screening_inputs(
            source_prompts_path=args.source_prompts,
            output_dir=args.output_dir,
            family_name=args.family_name,
            screening_spec_path=args.screening_spec_path,
        )
    else:
        result = run_family_screening(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
