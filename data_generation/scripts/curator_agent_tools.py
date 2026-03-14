from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def discover_run_artifacts(runs_root: Path) -> List[Dict[str, Path]]:
    results: List[Dict[str, Path]] = []
    for path in runs_root.rglob("dataset.json"):
        run_dir = path.parent
        full_log = run_dir / "full_log.json"
        validation_report = run_dir / "validation_report.json"
        if full_log.exists() and validation_report.exists():
            results.append(
                {
                    "run_dir": run_dir,
                    "dataset": path,
                    "full_log": full_log,
                    "validation_report": validation_report,
                }
            )
    return sorted(results, key=lambda row: str(row["run_dir"]))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build_pair_records(artifacts: Dict[str, Path]) -> List[Dict[str, object]]:
    dataset = _read_json(artifacts["dataset"])
    full_log = _read_json(artifacts["full_log"])
    model_id = (dataset.get("metadata") or {}).get("model_id") or "unknown"

    dataset_pairs_by_id = {
        pair.get("id"): pair
        for pair in dataset.get("pairs", [])
        if pair.get("id") is not None
    }

    pair_records: List[Dict[str, object]] = []
    for row in full_log:
        pair_id = row.get("pair_id")
        dataset_pair = dataset_pairs_by_id.get(pair_id, {})
        final_validation = row.get("final_validation") or row.get("judge_result") or {}
        dimension = (
            row.get("dimension")
            or row.get("expected_attribute")
            or ((dataset_pair.get("degradation") or {}).get("attribute"))
            or "unknown"
        )
        prompt = row.get("positive_prompt") or ((dataset_pair.get("positive") or {}).get("prompt")) or ""
        valid = bool(final_validation.get("valid"))
        failure = final_validation.get("failure")
        pair_records.append(
            {
                "pair_id": pair_id,
                "model_id": model_id,
                "dimension": dimension,
                "severity": row.get("severity") or ((dataset_pair.get("degradation") or {}).get("severity")),
                "positive_prompt": prompt,
                "valid": valid,
                "failure": failure,
                "success": bool(row.get("success")),
                "run_dir": str(artifacts["run_dir"]),
            }
        )
    return pair_records


def aggregate_failure_patterns(run_artifacts: Iterable[Dict[str, Path]]) -> Dict[str, object]:
    totals: Counter = Counter()
    by_model_dimension: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for artifacts in run_artifacts:
        for row in _build_pair_records(artifacts):
            failure = row.get("failure")
            if not failure:
                continue
            model_id = row.get("model_id") or "unknown"
            dimension = row.get("dimension") or "unknown"
            totals[failure] += 1
            by_model_dimension[model_id][dimension][failure] += 1

    return {
        "totals": dict(totals),
        "by_model_dimension": {
            model_id: {dimension: dict(counter) for dimension, counter in dims.items()}
            for model_id, dims in by_model_dimension.items()
        },
    }


def aggregate_failure_patterns_by_prompt(run_artifacts: Iterable[Dict[str, Path]]) -> Dict[str, object]:
    grouped: Dict[tuple, Dict[str, object]] = {}

    for artifacts in run_artifacts:
        for row in _build_pair_records(artifacts):
            failure = row.get("failure")
            if not failure:
                continue
            prompt = row.get("positive_prompt") or ""
            model_id = row.get("model_id") or "unknown"
            dimension = row.get("dimension") or "unknown"
            key = (prompt, model_id, dimension)
            bucket = grouped.setdefault(
                key,
                {
                    "prompt_text": prompt,
                    "model_id": model_id,
                    "dimension": dimension,
                    "failure_count": 0,
                    "failure_types": Counter(),
                },
            )
            bucket["failure_count"] += 1
            bucket["failure_types"][failure] += 1

    rows = [
        {
            "prompt_text": bucket["prompt_text"],
            "model_id": bucket["model_id"],
            "dimension": bucket["dimension"],
            "failure_count": bucket["failure_count"],
            "failure_types": dict(bucket["failure_types"]),
        }
        for bucket in grouped.values()
    ]
    rows.sort(key=lambda row: (-row["failure_count"], row["model_id"], row["dimension"], row["prompt_text"]))
    return {"rows": rows}


def build_blacklist_candidates(run_artifacts: Iterable[Dict[str, Path]], failure_threshold: int) -> Dict[str, object]:
    grouped: Dict[tuple, Dict[str, object]] = {}

    for artifacts in run_artifacts:
        for row in _build_pair_records(artifacts):
            failure = row.get("failure")
            if not failure:
                continue
            prompt = row.get("positive_prompt") or ""
            model_id = row.get("model_id") or "unknown"
            dimension = row.get("dimension") or "unknown"
            key = (prompt, model_id, dimension)
            bucket = grouped.setdefault(
                key,
                {
                    "prompt_text": prompt,
                    "model_id": model_id,
                    "dimension": dimension,
                    "failure_count": 0,
                    "failure_types": Counter(),
                },
            )
            bucket["failure_count"] += 1
            bucket["failure_types"][failure] += 1

    candidates = []
    for bucket in grouped.values():
        if bucket["failure_count"] < failure_threshold:
            continue
        candidates.append(
            {
                "prompt_text": bucket["prompt_text"],
                "model_id": bucket["model_id"],
                "dimension": bucket["dimension"],
                "failure_count": bucket["failure_count"],
                "failure_types": dict(bucket["failure_types"]),
                "candidate_reason": "repeated_failures_for_same_prompt_model_dimension",
            }
        )
    candidates.sort(key=lambda row: (-row["failure_count"], row["model_id"], row["dimension"], row["prompt_text"]))
    return {"failure_threshold": failure_threshold, "candidates": candidates}


def build_curation_decisions(
    run_artifacts: Iterable[Dict[str, Path]], blacklist: Dict[str, object]
) -> List[Dict[str, object]]:
    blacklist_keys = {
        (row["prompt_text"], row["model_id"], row["dimension"]) for row in blacklist.get("candidates", [])
    }
    decisions: List[Dict[str, object]] = []

    for artifacts in run_artifacts:
        for pair in _build_pair_records(artifacts):
            prompt = pair.get("positive_prompt") or ""
            model_id = pair.get("model_id") or "unknown"
            dimension = pair.get("dimension") or "unknown"
            valid = bool(pair.get("valid"))
            key = (prompt, model_id, dimension)
            if key in blacklist_keys:
                decision = "blacklist_candidate"
                reason = "prompt_repeatedly_failed_in_same_model_dimension"
            elif valid:
                decision = "keep"
                reason = "pair_valid"
            else:
                decision = "review"
                reason = pair.get("failure") or "pair_invalid"
            decisions.append(
                {
                    "pair_id": pair.get("pair_id"),
                    "model_id": model_id,
                    "dimension": dimension,
                    "decision": decision,
                    "reason": reason,
                    "positive_prompt": prompt,
                }
            )
    return decisions


def build_memory_stats(
    run_artifacts: Iterable[Dict[str, Path]], decisions: List[Dict[str, object]], blacklist: Dict[str, object]
) -> Dict[str, object]:
    runs = 0
    pairs = 0
    invalid_pairs = 0
    for artifacts in run_artifacts:
        runs += 1
        rows = _build_pair_records(artifacts)
        pairs += len(rows)
        invalid_pairs += sum(1 for pair in rows if not bool(pair.get("valid")))

    return {
        "runs": runs,
        "pairs": pairs,
        "invalid_pairs": invalid_pairs,
        "blacklist_candidates": len(blacklist.get("candidates", [])),
        "decision_counts": dict(Counter(row["decision"] for row in decisions)),
    }
