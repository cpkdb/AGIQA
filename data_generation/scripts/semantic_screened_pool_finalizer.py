#!/usr/bin/env python3
"""Finalize completed semantic-screened family runs into reusable pool artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prompt_pool_agent_tools as prompt_pool_tools
from sd35_turbo_pool_builder import (
    DEFAULT_GLOBAL_MAX_TOKENS,
    DEFAULT_STRICT_MAX_TOKENS,
    STRICT_DIMENSIONS,
    build_clip_token_counter,
    filter_records_by_clip_tokens,
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _dedupe_by_prompt(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        deduped.append(dict(row))
    return deduped


def _load_pass_rows(results_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows = _read_jsonl(results_path)
    stats = {"total": len(rows), "pass": 0, "fail": 0}
    passes: List[Dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        if label == "pass":
            stats["pass"] += 1
            passes.append(dict(row))
        else:
            stats["fail"] += 1
    return _dedupe_by_prompt(passes), stats


def _resolve_family_result_rows(run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    direct_results = run_dir / "family_screening_results.jsonl"
    if direct_results.exists():
        return _load_pass_rows(direct_results)

    shard_rows: List[Dict[str, Any]] = []
    found_shard = False
    for shard_dir in sorted(run_dir.glob("shard_*")):
        results_path = shard_dir / "family_screening_results.jsonl"
        if not results_path.exists():
            continue
        found_shard = True
        shard_rows.extend(_read_jsonl(results_path))
    if not found_shard:
        raise FileNotFoundError(f"No family_screening_results.jsonl found under {run_dir}")

    stats = {"total": len(shard_rows), "pass": 0, "fail": 0}
    passes: List[Dict[str, Any]] = []
    for row in shard_rows:
        label = str(row.get("label", "")).strip().lower()
        if label == "pass":
            stats["pass"] += 1
            passes.append(dict(row))
        else:
            stats["fail"] += 1
    return _dedupe_by_prompt(passes), stats


def _family_to_dimensions(screening_spec: Mapping[str, Any]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for dimension, override in screening_spec.get("dimension_overrides", {}).items():
        family = override.get("shared_pool_family")
        if not family:
            continue
        mapping.setdefault(family, []).append(dimension)
    return {family: sorted(dimensions) for family, dimensions in mapping.items()}


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


def _has_any_base_tag(record: Dict[str, Any], base_tags: Iterable[str]) -> bool:
    tags = set(_normalize_tags(record))
    signature = _normalize_signature(record)
    for tag in base_tags:
        if tag in tags or bool(signature.get(tag)):
            return True
    return False


def _build_rule_only_dimension_rows(
    *,
    source_records: Iterable[Dict[str, Any]],
    base_tags: Iterable[str],
    dimension: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for record in source_records:
        prompt = str(record.get("prompt", "")).strip()
        if not prompt or prompt in seen:
            continue
        if not _has_any_base_tag(record, base_tags):
            continue
        seen.add(prompt)
        row = dict(record)
        row["dimension"] = dimension
        row["source_pool_mode"] = "rule_recall_only"
        rows.append(row)
    return rows


def finalize_common_screened_pools(
    *,
    family_run_dirs: Mapping[str, str],
    output_root: str,
    screening_spec_path: str = str(prompt_pool_tools.DEFAULT_SCREENING_SPEC_PATH),
    build_targets_path: str = str(prompt_pool_tools.DEFAULT_BUILD_TARGETS_PATH),
    common_source_prompts_path: str | None = None,
) -> Dict[str, str]:
    screening_spec = prompt_pool_tools.load_prompt_pool_screening_spec(Path(screening_spec_path))
    build_targets = prompt_pool_tools.load_prompt_pool_build_targets(Path(build_targets_path))
    family_to_dims = _family_to_dimensions(screening_spec)

    output_dir = Path(output_root).resolve()
    family_dir = output_dir / build_targets["shared_pool_family_outputs"]["common"]["directory_name"]
    dimension_dir = output_dir / build_targets["dimension_subpool_outputs"]["common"]["directory_name"]
    family_dir.mkdir(parents=True, exist_ok=True)
    dimension_dir.mkdir(parents=True, exist_ok=True)

    common_family_index: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "schema_version": "1.0",
        "families": {},
    }
    common_dimension_index: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "schema_version": "1.0",
        "dimensions": {},
    }

    family_rows: Dict[str, List[Dict[str, Any]]] = {}

    for family_name, run_dir_str in family_run_dirs.items():
        run_dir = Path(run_dir_str).resolve()
        pass_rows, stats = _resolve_family_result_rows(run_dir)
        family_rows[family_name] = pass_rows

        family_file = family_dir / f"{family_name}.jsonl"
        _write_jsonl(family_file, pass_rows)
        target_dimensions = family_to_dims.get(family_name, [])
        common_family_index["families"][family_name] = {
            "filename": str(family_file.resolve()),
            "count": len(pass_rows),
            "source_run_dir": str(run_dir),
            "total_results": stats["total"],
            "pass_count": stats["pass"],
            "fail_count": stats["fail"],
            "target_dimensions": target_dimensions,
        }

    for family_name, dimensions in family_to_dims.items():
        rows = family_rows.get(family_name, [])
        for dimension in dimensions:
            dimension_file = dimension_dir / f"{dimension}.jsonl"
            _write_jsonl(dimension_file, rows)
            common_dimension_index["dimensions"][dimension] = {
                "filename": str(dimension_file.resolve()),
                "count": len(rows),
                "source_family": family_name,
            }

    if common_source_prompts_path:
        source_records = _read_jsonl(Path(common_source_prompts_path).resolve())
        for dimension, override in screening_spec.get("dimension_overrides", {}).items():
            if override.get("builder_mode") != "rule_recall_only":
                continue
            base_tags = list(override.get("base_tags", []))
            rows = _build_rule_only_dimension_rows(
                source_records=source_records,
                base_tags=base_tags,
                dimension=dimension,
            )
            dimension_file = dimension_dir / f"{dimension}.jsonl"
            _write_jsonl(dimension_file, rows)
            common_dimension_index["dimensions"][dimension] = {
                "filename": str(dimension_file.resolve()),
                "count": len(rows),
                "source_family": "rule_recall_only",
            }

    family_index_path = family_dir / build_targets["shared_pool_family_outputs"]["common"]["index_name"]
    dimension_index_path = dimension_dir / build_targets["dimension_subpool_outputs"]["common"]["index_name"]
    _write_json(family_index_path, common_family_index)
    _write_json(dimension_index_path, common_dimension_index)

    return {
        "common_family_dir": str(family_dir),
        "common_family_index": str(family_index_path),
        "common_dimension_dir": str(dimension_dir),
        "common_dimension_index": str(dimension_index_path),
    }


def finalize_turbo_derived_screened_pools(
    *,
    common_family_dir: str,
    common_dimension_dir: str,
    output_root: str,
    build_targets_path: str = str(prompt_pool_tools.DEFAULT_BUILD_TARGETS_PATH),
    measure_tokens: Callable[[str], int],
    global_max_tokens: int = DEFAULT_GLOBAL_MAX_TOKENS,
    strict_max_tokens: int = DEFAULT_STRICT_MAX_TOKENS,
) -> Dict[str, str]:
    build_targets = prompt_pool_tools.load_prompt_pool_build_targets(Path(build_targets_path))

    common_family_root = Path(common_family_dir).resolve()
    common_dimension_root = Path(common_dimension_dir).resolve()
    common_family_index = _read_json(common_family_root / "index.json")
    common_dimension_index = _read_json(common_dimension_root / "index.json")

    output_dir = Path(output_root).resolve()
    turbo_family_dir = output_dir / build_targets["shared_pool_family_outputs"]["turbo"]["directory_name"]
    turbo_dimension_dir = output_dir / build_targets["dimension_subpool_outputs"]["turbo"]["directory_name"]
    turbo_family_dir.mkdir(parents=True, exist_ok=True)
    turbo_dimension_dir.mkdir(parents=True, exist_ok=True)

    turbo_family_index: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "schema_version": "1.0",
        "families": {},
    }
    turbo_dimension_index: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "schema_version": "1.0",
        "dimensions": {},
    }

    for family_name, meta in common_family_index.get("families", {}).items():
        records = _read_jsonl(Path(meta["filename"]))
        filtered, summary = filter_records_by_clip_tokens(
            records,
            max_tokens=global_max_tokens,
            measure_tokens=measure_tokens,
        )
        out_path = turbo_family_dir / f"{family_name}.jsonl"
        _write_jsonl(out_path, filtered)
        turbo_family_index["families"][family_name] = {
            "filename": str(out_path.resolve()),
            "count": len(filtered),
            "source_family_file": meta["filename"],
            "target_dimensions": list(meta.get("target_dimensions", [])),
            "token_filter_summary": summary,
            "max_tokens": global_max_tokens,
        }

    for dimension, meta in common_dimension_index.get("dimensions", {}).items():
        records = _read_jsonl(Path(meta["filename"]))
        threshold = strict_max_tokens if dimension in STRICT_DIMENSIONS else global_max_tokens
        filtered, summary = filter_records_by_clip_tokens(
            records,
            max_tokens=threshold,
            measure_tokens=measure_tokens,
        )
        out_path = turbo_dimension_dir / f"{dimension}.jsonl"
        _write_jsonl(out_path, filtered)
        turbo_dimension_index["dimensions"][dimension] = {
            "filename": str(out_path.resolve()),
            "count": len(filtered),
            "source_dimension_file": meta["filename"],
            "source_family": meta.get("source_family"),
            "token_filter_summary": summary,
            "max_tokens": threshold,
        }

    family_index_path = turbo_family_dir / build_targets["shared_pool_family_outputs"]["turbo"]["index_name"]
    dimension_index_path = turbo_dimension_dir / build_targets["dimension_subpool_outputs"]["turbo"]["index_name"]
    _write_json(family_index_path, turbo_family_index)
    _write_json(dimension_index_path, turbo_dimension_index)

    return {
        "turbo_family_dir": str(turbo_family_dir),
        "turbo_family_index": str(family_index_path),
        "turbo_dimension_dir": str(turbo_dimension_dir),
        "turbo_dimension_index": str(dimension_index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize semantic-screened shared family pools into reusable outputs.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--human_run_dir")
    parser.add_argument("--object_run_dir")
    parser.add_argument("--multi_run_dir")
    parser.add_argument("--common_family_dir")
    parser.add_argument("--common_dimension_dir")
    parser.add_argument("--screening_spec_path", default=str(prompt_pool_tools.DEFAULT_SCREENING_SPEC_PATH))
    parser.add_argument("--build_targets_path", default=str(prompt_pool_tools.DEFAULT_BUILD_TARGETS_PATH))
    parser.add_argument("--common_source_prompts")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--tokenizer2_path")
    parser.add_argument("--global_max_tokens", type=int, default=DEFAULT_GLOBAL_MAX_TOKENS)
    parser.add_argument("--strict_max_tokens", type=int, default=DEFAULT_STRICT_MAX_TOKENS)
    parser.add_argument("--skip_turbo", action="store_true")
    args = parser.parse_args()

    if args.common_family_dir and args.common_dimension_dir:
        common_outputs = {
            "common_family_dir": args.common_family_dir,
            "common_dimension_dir": args.common_dimension_dir,
        }
    else:
        family_run_dirs = {
            "human_full_body_realistic": args.human_run_dir,
            "structured_object_primary": args.object_run_dir,
            "multi_object_reference": args.multi_run_dir,
        }
        missing = [name for name, value in family_run_dirs.items() if not value]
        if missing:
            raise SystemExit(f"Missing family run dirs: {', '.join(missing)}")
        common_outputs = finalize_common_screened_pools(
            family_run_dirs=family_run_dirs,
            output_root=args.output_root,
            screening_spec_path=args.screening_spec_path,
            build_targets_path=args.build_targets_path,
            common_source_prompts_path=args.common_source_prompts,
        )

    outputs: Dict[str, Any] = dict(common_outputs)

    if not args.skip_turbo:
        if not args.tokenizer_path or not args.tokenizer2_path:
            raise SystemExit("--tokenizer_path and --tokenizer2_path are required unless --skip_turbo is set")
        measure_tokens = build_clip_token_counter(args.tokenizer_path, args.tokenizer2_path)
        turbo_outputs = finalize_turbo_derived_screened_pools(
            common_family_dir=common_outputs["common_family_dir"],
            common_dimension_dir=common_outputs["common_dimension_dir"],
            output_root=args.output_root,
            build_targets_path=args.build_targets_path,
            measure_tokens=measure_tokens,
            global_max_tokens=args.global_max_tokens,
            strict_max_tokens=args.strict_max_tokens,
        )
        outputs.update(turbo_outputs)

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
