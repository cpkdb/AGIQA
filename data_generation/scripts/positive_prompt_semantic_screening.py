#!/usr/bin/env python3
"""
LLM semantic screening for sampled positive prompts in dimension subpools.

This script:
1) Selects dimensions (default: llm_semantic_screening_after_merge=true in plan)
2) Samples prompts per dimension from dimension subpools
3) Calls an OpenAI-compatible API in batches for pass/fail/uncertain judgement
4) Writes JSONL judgements and summary metrics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN_PATH = SCRIPT_DIR.parent / "data" / "llm_backfill_plan_all_dimensions_v1.json"
DEFAULT_SUBPOOL_INDEX_PATH = Path(
    "/root/autodl-tmp/AGIQA/data/prompt_sources_workspace"
    "/backfill_merge_runs/all_dimensions_v1_full"
    "/dimension_subpools/index.json"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "data" / "semantic_screening_runs"


@dataclass
class PromptSample:
    dimension: str
    sample_id: str
    prompt: str


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def _load_llm_config(path: Path) -> Dict:
    def _coerce_scalar(raw: str):
        value = raw.strip().strip("'").strip('"')
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.lower() in {"null", "none"}:
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _load_simple_yaml_fallback() -> Dict:
        payload: Dict = {}
        current_section: Optional[str] = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" ") and line.rstrip().endswith(":"):
                current_section = line.rstrip()[:-1]
                payload[current_section] = {}
                continue
            if current_section is None:
                continue
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            payload[current_section][key.strip()] = _coerce_scalar(value)
        return payload

    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_simple_yaml_fallback()

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _create_openai_client(config: Dict):
    from openai import OpenAI  # type: ignore

    def _normalize_base_url(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return raw
        value = raw.strip().rstrip("/")
        for suffix in ("/chat/completions", "/responses", "/models"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        return value

    llm = config.get("llm", {})
    api_key = llm.get("api_key")
    if not api_key:
        raise ValueError("llm.api_key is required")

    base_url = _normalize_base_url(llm.get("api_base"))
    timeout = llm.get("timeout")
    if base_url and timeout:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=float(timeout))
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    if timeout:
        return OpenAI(api_key=api_key, timeout=float(timeout))
    return OpenAI(api_key=api_key)


def _dimensions_from_plan(plan_path: Path) -> List[str]:
    plan = _read_json(plan_path)
    dims = [
        item["dimension"]
        for item in plan.get("dimensions", [])
        if item.get("llm_semantic_screening_after_merge") is True
    ]
    if not dims:
        raise ValueError("No dimensions with llm_semantic_screening_after_merge=true")
    return dims


def _resolve_dimensions(args, plan_path: Path) -> List[str]:
    if args.dimensions:
        return [item.strip() for item in args.dimensions.split(",") if item.strip()]
    return _dimensions_from_plan(plan_path)


def _sample_dimension_prompts(
    subpool_file: Path,
    dimension: str,
    sample_size: int,
    seed: int,
) -> List[PromptSample]:
    rows = _read_jsonl(subpool_file)
    prompts = [(row.get("prompt") or "").strip() for row in rows]
    prompts = [p for p in prompts if p]
    unique = list(dict.fromkeys(prompts))
    rng = random.Random(f"{seed}:{dimension}")
    rng.shuffle(unique)
    selected = unique[: min(sample_size, len(unique))]
    samples: List[PromptSample] = []
    for idx, prompt in enumerate(selected, start=1):
        sample_id = f"{dimension}-{idx:05d}-{_prompt_hash(prompt)}"
        samples.append(PromptSample(dimension=dimension, sample_id=sample_id, prompt=prompt))
    return samples


def _build_system_prompt(policy: str) -> str:
    if policy == "lenient":
        return (
            "You are a pragmatic data quality reviewer for text-to-image positive prompts. "
            "Judge whether each prompt is likely to generate images that contain enough semantic signal for downstream "
            "degradation on the target dimension. "
            "The prompt does not need explicit or strict wording, but must not already describe defects/degradation. "
            "Return JSON only."
        )
    return (
        "You are a strict data quality reviewer for text-to-image positive prompts. "
        "Judge whether each prompt is a suitable positive anchor for the target dimension's semantic precondition. "
        "Do not reward degraded/defective descriptions. "
        "Return JSON only."
    )


def _build_user_prompt(dimension: str, batch: List[PromptSample], policy: str) -> str:
    if policy == "lenient":
        rule_lines = [
            "Decision rule (LENIENT):",
            "- pass: prompt likely contains enough semantic cues so generated images can expose this dimension for degradation.",
            "- fail: prompt is clearly irrelevant/impossible for this dimension or already describes degraded output.",
            "- uncertain: borderline/ambiguous cases.",
        ]
    else:
        rule_lines = [
            "Decision rule (STRICT):",
            "- pass: prompt clearly contains semantic elements needed to evaluate this dimension as a clean positive anchor.",
            "- fail: prompt clearly lacks required elements or is unsuitable/noisy for this dimension.",
            "- uncertain: ambiguous or insufficiently specified.",
        ]

    lines = [
        f"Target dimension: {dimension}",
        "Task: For each prompt, return label in {pass, fail, uncertain}, confidence in [0,1], and short reason.",
        *rule_lines,
        "Output JSON schema:",
        '{"dimension":"...","results":[{"sample_id":"...","label":"pass|fail|uncertain","confidence":0.0,"reason":"..."}]}',
        "Prompts:",
    ]
    for item in batch:
        lines.append(f'- sample_id="{item.sample_id}" prompt="{item.prompt}"')
    return "\n".join(lines)


def _extract_json_obj(raw: str) -> Dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        try:
            arr = json.loads(text[start_arr : end_arr + 1])
            if isinstance(arr, list):
                return {"results": arr}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Unable to parse JSON object from model response")


def _normalize_label(label: str) -> str:
    value = (label or "").strip().lower()
    if value in {"pass", "ok", "yes"}:
        return "pass"
    if value in {"fail", "no", "reject"}:
        return "fail"
    return "uncertain"


def _is_kept(label: str, policy: str) -> bool:
    if policy == "lenient":
        return label in {"pass", "uncertain"}
    return label == "pass"


def _call_llm_with_retry(
    client,
    llm_cfg: Dict,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, Dict]:
    max_retries = int(llm_cfg.get("max_retries", 3))
    retry_delay = float(llm_cfg.get("retry_delay", 2))
    reasoning_effort = llm_cfg.get("reasoning_effort")
    if reasoning_effort is None and str(model).startswith("gpt-5"):
        reasoning_effort = "minimal"
    include_reasoning_effort = reasoning_effort is not None
    max_tokens = int(llm_cfg.get("max_tokens", 1200))
    if str(model).startswith("gpt-5") and max_tokens < 1200:
        max_tokens = 1200

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        request_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": llm_cfg.get("temperature", 0.0),
            "max_tokens": max_tokens,
            "top_p": llm_cfg.get("top_p", 1.0),
        }
        if include_reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort
        try:
            resp = client.chat.completions.create(**request_kwargs)
            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            usage_payload = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            return content, usage_payload
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc).lower()
            if include_reasoning_effort and ("reasoning_effort" in msg or "unknown parameter" in msg):
                include_reasoning_effort = False
                continue
            if attempt >= max_retries:
                break
            sleep_sec = retry_delay * (attempt + 1)
            time.sleep(sleep_sec)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def run_screening(args) -> Dict:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (DEFAULT_OUTPUT_ROOT / f"run_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_config_path = Path(args.llm_config).resolve()
    config = _load_llm_config(llm_config_path)
    llm_cfg = config.get("llm", {})
    model = args.model_override or llm_cfg.get("model", "gpt-4o")
    client = _create_openai_client(config)

    plan_path = Path(args.plan_path).resolve()
    dimensions = _resolve_dimensions(args, plan_path=plan_path)
    if args.max_dimensions is not None:
        dimensions = dimensions[: args.max_dimensions]

    subpool_index = _read_json(Path(args.subpool_index).resolve())
    subpool_dir = Path(args.subpool_index).resolve().parent
    dim_meta = subpool_index.get("dimensions", {})

    results_jsonl = output_dir / "screening_results.jsonl"
    raw_jsonl = output_dir / "raw_llm_responses.jsonl"
    summary_path = output_dir / "screening_summary.json"
    manifest_path = output_dir / "screening_manifest.json"

    already_done = set()
    if args.resume and results_jsonl.exists():
        for row in _read_jsonl(results_jsonl):
            sid = row.get("sample_id")
            if sid:
                already_done.add(sid)

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "llm_config_path": str(llm_config_path),
        "model": model,
        "dimensions": dimensions,
        "samples_per_dimension": args.samples_per_dimension,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "subpool_index": str(Path(args.subpool_index).resolve()),
        "resume": bool(args.resume),
        "screening_policy": args.screening_policy,
    }
    _write_json(manifest_path, manifest)

    system_prompt = _build_system_prompt(policy=args.screening_policy)
    request_counter = 0
    by_dimension_counts: Dict[str, Dict[str, int]] = {}
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for dimension in dimensions:
        if dimension not in dim_meta:
            raise KeyError(f"Dimension not found in subpool index: {dimension}")
        subpool_file = subpool_dir / dim_meta[dimension]["filename"]
        samples = _sample_dimension_prompts(
            subpool_file=subpool_file,
            dimension=dimension,
            sample_size=args.samples_per_dimension,
            seed=args.seed,
        )
        if args.resume and already_done:
            samples = [item for item in samples if item.sample_id not in already_done]

        if dimension not in by_dimension_counts:
            by_dimension_counts[dimension] = {"pass": 0, "fail": 0, "uncertain": 0, "keep": 0, "total": 0}

        for i in range(0, len(samples), args.batch_size):
            batch = samples[i : i + args.batch_size]
            if not batch:
                continue
            request_counter += 1
            user_prompt = _build_user_prompt(
                dimension=dimension,
                batch=batch,
                policy=args.screening_policy,
            )
            raw_text, usage = _call_llm_with_retry(
                client=client,
                llm_cfg=llm_cfg,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            for k in usage_totals:
                val = usage.get(k)
                if isinstance(val, int):
                    usage_totals[k] += val

            parse_error = None
            try:
                parsed = _extract_json_obj(raw_text)
                items = parsed.get("results", [])
            except Exception as exc:  # noqa: BLE001
                parse_error = str(exc)
                items = []
            by_id = {item.sample_id: item for item in batch}
            out_rows: List[Dict] = []

            for result in items:
                sid = str(result.get("sample_id", "")).strip()
                if sid not in by_id:
                    continue
                label = _normalize_label(str(result.get("label", "uncertain")))
                confidence = result.get("confidence", 0.0)
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))
                reason = str(result.get("reason", "")).strip()
                sample = by_id[sid]
                out_rows.append(
                    {
                        "dimension": dimension,
                        "sample_id": sid,
                        "prompt": sample.prompt,
                        "label": label,
                        "keep_by_policy": _is_kept(label=label, policy=args.screening_policy),
                        "confidence": confidence,
                        "reason": reason,
                        "request_id": request_counter,
                        "model": model,
                    }
                )

            # Fallback: for missing ids in model output, mark uncertain
            returned_ids = {row["sample_id"] for row in out_rows}
            for sid, sample in by_id.items():
                if sid in returned_ids:
                    continue
                out_rows.append(
                    {
                        "dimension": dimension,
                        "sample_id": sid,
                        "prompt": sample.prompt,
                        "label": "uncertain",
                        "keep_by_policy": _is_kept(label="uncertain", policy=args.screening_policy),
                        "confidence": 0.0,
                        "reason": "missing_in_model_output",
                        "request_id": request_counter,
                        "model": model,
                    }
                )

            _append_jsonl(results_jsonl, out_rows)
            _append_jsonl(
                raw_jsonl,
                [
                    {
                        "dimension": dimension,
                        "request_id": request_counter,
                        "batch_size": len(batch),
                        "raw_content": raw_text,
                        "usage": usage,
                        "parse_error": parse_error,
                    }
                ],
            )

            for row in out_rows:
                by_dimension_counts[dimension]["total"] += 1
                by_dimension_counts[dimension][row["label"]] += 1
                if row.get("keep_by_policy"):
                    by_dimension_counts[dimension]["keep"] += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    summary = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "model": model,
        "samples_per_dimension": args.samples_per_dimension,
        "batch_size": args.batch_size,
        "request_count": request_counter,
        "usage_totals": usage_totals,
        "screening_policy": args.screening_policy,
        "dimensions": {},
    }

    for dimension, stats in by_dimension_counts.items():
        total = stats["total"] or 1
        summary["dimensions"][dimension] = {
            **stats,
            "keep_rate": round(stats["keep"] / total, 4),
            "pass_rate": round(stats["pass"] / total, 4),
            "fail_rate": round(stats["fail"] / total, 4),
            "uncertain_rate": round(stats["uncertain"] / total, 4),
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
    parser = argparse.ArgumentParser(description="LLM semantic screening for sampled dimension subpools.")
    parser.add_argument("--llm_config", type=str, required=True)
    parser.add_argument("--plan_path", type=str, default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--subpool_index", type=str, default=str(DEFAULT_SUBPOOL_INDEX_PATH))
    parser.add_argument("--dimensions", type=str, default=None, help="Comma-separated dimensions.")
    parser.add_argument("--max_dimensions", type=int, default=None)
    parser.add_argument("--samples_per_dimension", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--sleep_seconds", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--screening_policy",
        type=str,
        default="lenient",
        choices=["lenient", "strict"],
        help="lenient: keep pass+uncertain; strict: keep pass only.",
    )
    parser.add_argument("--model_override", type=str, default=None, help="Optional model override for this run.")
    args = parser.parse_args()

    result = run_screening(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
