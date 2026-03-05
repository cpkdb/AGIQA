#!/usr/bin/env python3
"""
Prepare and optionally execute LLM requests for positive prompt backfill.

This script is the execution layer on top of:
- positive_prompt_backfill_renderer.py
- positive_prompt_backfill_summary.py

By default it writes a request manifest only. When `--execute` is provided and a
valid LLM config path is supplied, it will call an OpenAI-compatible endpoint and
write generated prompts to JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "positive_prompt_backfill_runs"
SUMMARY_PATH = SCRIPT_DIR / "positive_prompt_backfill_summary.py"
RENDERER_PATH = SCRIPT_DIR / "positive_prompt_backfill_renderer.py"


def _load_module(module_path: Path, module_name: str):
    spec = spec_from_file_location(module_name, module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_summary_module():
    return _load_module(SUMMARY_PATH, "positive_prompt_backfill_summary")


def _load_renderer_module():
    return _load_module(RENDERER_PATH, "positive_prompt_backfill_renderer")


def _chunk_count(total: int, chunk_size: int) -> List[int]:
    if total <= 0:
        return []
    chunks: List[int] = []
    remaining = total
    while remaining > 0:
        current = min(chunk_size, remaining)
        chunks.append(current)
        remaining -= current
    return chunks


def _select_dimensions(
    dimensions: Optional[List[str]] = None,
    max_dimensions: Optional[int] = None,
) -> List[Dict]:
    summary_module = _load_summary_module()
    if dimensions:
        summary = summary_module.build_summary()
        selected = []
        for dimension in dimensions:
            if dimension not in summary["dimensions"]:
                raise KeyError(f"Unknown active dimension: {dimension}")
            selected.append(summary["dimensions"][dimension])
        selected.sort(key=lambda item: (item["priority"], -item["target_new_prompts"], item["dimension"]))
        return selected

    queue = summary_module.build_priority_queue(limit=max_dimensions)
    return list(queue)


def _build_user_prompt(job: Dict) -> str:
    template_family = job["template_family"]
    structure_instruction = (
        "Use clear subject, scene, and attribute wording that keeps the target structure visually salient."
    )
    positive_anchor_instruction = (
        "The prompt must not already exhibit the target degradation. It should describe a clean positive anchor "
        "where the relevant structure is clearly present, natural, and intact."
    )

    template_specific = {
        "text_literal_template": (
            "Each prompt must include an explicit text carrier and quoted readable literal text. "
            "Prefer signs, labels, menus, packaging, covers, or screens. "
            "The text itself must be correct and clearly readable, with no spelling mistakes, garbling, or malformed typography."
        ),
        "hands_visible_template": (
            "Each prompt must make hands clearly visible and engaged in a concrete action or interaction. "
            "Prefer holding, pointing, typing, drawing, cooking, or similar hand-centric actions. "
            "Hands must appear natural, complete, and anatomically normal."
        ),
        "portrait_face_template": (
            "Each prompt must present a clearly visible face with portrait or close-up framing, plus an explicit facial expression. "
            "The face should remain natural, symmetric, and cleanly rendered."
        ),
        "full_body_human_template": (
            "Each prompt must make the full human body or whole-body pose clearly visible, with a natural but explicit pose description. "
            "Body proportions and pose must remain physically plausible and anatomically normal."
        ),
        "animal_focus_template": (
            "Each prompt must keep one clear animal subject visually dominant with low background ambiguity. "
            "The animal anatomy must remain natural and biologically plausible."
        ),
        "structured_object_template": (
            "Each prompt must feature a clearly bounded man-made object with stable structure and recognizable parts. "
            "Object form should remain coherent, complete, and logically assembled."
        ),
        "relation_count_template": (
            "Each prompt must include explicit countable entities or quantity cues, with counts or count-sensitive wording. "
            "The described counts, sizes, and object identities should remain internally consistent."
        ),
        "multi_object_layout_template": (
            "Each prompt must include multiple objects or a structured spatial layout only when needed for the target relation or composition. "
            "Spatial arrangement should remain coherent, non-overlapping unless naturally intended, and physically plausible."
        ),
        "generic_scene_context_template": (
            "Each prompt must include enough scene context to define environment logic while keeping the key subject and relations attributable. "
            "Scene context must remain logically consistent and visually coherent."
        ),
        "lighting_exposure_anchor_template": (
            "Each prompt should include semantically clean lighting context without introducing style-heavy quality booster language. "
            "Lighting and exposure should remain natural and well-controlled."
        ),
        "lighting_anchor_template": (
            "Each prompt should include a clear light source or lighting geometry that remains semantically natural. "
            "Lighting direction, shadows, and illumination cues should remain internally consistent."
        ),
        "color_tone_anchor_template": (
            "Each prompt should include stable color-bearing subjects or palettes that are easy to preserve as positive anchors. "
            "Colors should remain believable, harmonious, and semantically appropriate."
        ),
        "generic_quality_anchor_template": (
            "Each prompt should be a clean, semantically natural positive anchor with readable structure and minimal style noise. "
            "Do not imply blur, distortion, bad composition, artifacts, or other quality defects."
        ),
    }.get(template_family, structure_instruction)

    return (
        f"Generate {job['requested_prompts']} high-quality positive prompts for the macro bucket "
        f"'{job['macro_bucket']}' that support the assigned downstream contrastive check. "
        f"Template family: {template_family}. "
        f"{positive_anchor_instruction} "
        f"{template_specific} "
        "Return only a JSON array of prompt strings. "
        "Keep prompts semantically natural, varied, and usable as positive anchors."
    )


def build_request_plan(
    dimensions: Optional[List[str]] = None,
    max_dimensions: Optional[int] = None,
    batch_size_prompts_per_request: int = 5,
    max_requests: Optional[int] = None,
) -> Dict:
    if batch_size_prompts_per_request <= 0:
        raise ValueError("batch_size_prompts_per_request must be positive")

    renderer = _load_renderer_module()
    selected_dimensions = _select_dimensions(dimensions=dimensions, max_dimensions=max_dimensions)

    jobs: List[Dict] = []
    request_counter = 0
    total_target_prompts = 0

    for item in selected_dimensions:
        rendered = renderer.render_generation_prompt(item["dimension"])
        dimension_plan = rendered["dimension_plan"]
        total_target_prompts += item["target_new_prompts"]

        for macro_bucket, total_bucket_prompts in rendered["scene_mix"].items():
            for chunk_index, requested_prompts in enumerate(
                _chunk_count(total_bucket_prompts, batch_size_prompts_per_request),
                start=1,
            ):
                if max_requests is not None and request_counter >= max_requests:
                    break
                request_counter += 1
                rendered_job = renderer.render_job_prompt(
                    dimension=item["dimension"],
                    macro_bucket=macro_bucket,
                    requested_prompts=requested_prompts,
                )
                jobs.append(
                    {
                        "request_id": request_counter,
                        "dimension": item["dimension"],
                        "priority": item["priority"],
                        "tier": item["tier"],
                        "coverage_mode": item["coverage_mode"],
                        "macro_bucket": macro_bucket,
                        "requested_prompts": requested_prompts,
                        "chunk_index": chunk_index,
                        "template_family": dimension_plan["template_family"],
                        "system_prompt": rendered_job["system_prompt"],
                        "user_prompt": _build_user_prompt(
                            {
                                "dimension": item["dimension"],
                                "macro_bucket": macro_bucket,
                                "requested_prompts": requested_prompts,
                                "template_family": dimension_plan["template_family"],
                            }
                        ),
                    }
                )
            if max_requests is not None and request_counter >= max_requests:
                break
        if max_requests is not None and request_counter >= max_requests:
            break

    return {
        "schema_version": "1.0",
        "dimension_count": len({job["dimension"] for job in jobs}),
        "selected_dimensions": [item["dimension"] for item in selected_dimensions],
        "total_target_prompts": total_target_prompts,
        "batch_size_prompts_per_request": batch_size_prompts_per_request,
        "total_requests": len(jobs),
        "jobs": jobs,
    }


def write_request_plan(
    output_dir: str,
    dimensions: Optional[List[str]] = None,
    max_dimensions: Optional[int] = None,
    batch_size_prompts_per_request: int = 5,
    max_requests: Optional[int] = None,
) -> Dict[str, str]:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    request_plan = build_request_plan(
        dimensions=dimensions,
        max_dimensions=max_dimensions,
        batch_size_prompts_per_request=batch_size_prompts_per_request,
        max_requests=max_requests,
    )

    manifest_path = output_root / "request_plan.json"
    manifest_path.write_text(
        json.dumps(request_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"manifest_path": str(manifest_path)}


def _load_llm_config(config_path: str) -> Dict:
    path = Path(config_path).resolve()
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    def _coerce_scalar(raw: str):
        raw = raw.strip()
        if not raw:
            return ""
        if raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            return raw[1:-1]
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

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
    except ImportError as exc:
        return _load_simple_yaml_fallback()

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _create_openai_client(config: Dict):
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("openai package is required for --execute mode") from exc

    llm = config.get("llm", {})
    api_key = llm.get("api_key")
    if not api_key:
        raise ValueError("llm.api_key is required for execute mode")
    base_url = llm.get("api_base")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _parse_prompt_list(raw_text: str, expected_count: int) -> List[str]:
    raw_text = raw_text.strip()
    prompts: List[str] = []

    def _extract_json_array_candidate(text: str) -> Optional[str]:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    candidate_payloads: List[str] = [raw_text]
    extracted = _extract_json_array_candidate(raw_text)
    if extracted and extracted != raw_text:
        candidate_payloads.append(extracted)

    for candidate in candidate_payloads:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                prompts = [str(item).strip() for item in parsed if str(item).strip()]
                if prompts:
                    break
        except json.JSONDecodeError:
            continue

    if not prompts:
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("```"):
                continue
            line = line.lstrip("-").strip()
            if line and line[0:2].isdigit() and ". " in line:
                line = line.split(". ", 1)[1].strip()
            prompts.append(line)

    return prompts[:expected_count]


def _build_completion_request_kwargs(job: Dict, llm: Dict, model: str) -> Dict:
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": job["system_prompt"]},
            {"role": "user", "content": job["user_prompt"]},
        ],
        "temperature": llm.get("temperature", 0.7),
        "max_tokens": llm.get("max_tokens", 500),
        "top_p": llm.get("top_p", 0.95),
    }

    reasoning_effort = llm.get("reasoning_effort")
    if reasoning_effort is None and model.startswith("gpt-5"):
        reasoning_effort = "minimal"
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort

    return request_kwargs


def execute_request_plan(
    request_plan: Dict,
    llm_config_path: str,
    output_dir: str,
    sleep_seconds: float = 0.0,
) -> Dict[str, str]:
    config = _load_llm_config(llm_config_path)
    client = _create_openai_client(config)
    llm = config.get("llm", {})
    model = llm.get("model", "gpt-4o")

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "request_plan.json"
    generated_path = output_root / "generated_positive_prompts.jsonl"
    raw_responses_path = output_root / "raw_llm_responses.jsonl"
    execution_summary_path = output_root / "execution_summary.json"

    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(request_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    total_generated = 0
    total_requests = 0

    with (
        generated_path.open("w", encoding="utf-8") as generated_handle,
        raw_responses_path.open("w", encoding="utf-8") as raw_handle,
    ):
        for job in request_plan["jobs"]:
            total_requests += 1
            response = client.chat.completions.create(
                **_build_completion_request_kwargs(job=job, llm=llm, model=model)
            )
            content = response.choices[0].message.content or ""
            prompts = _parse_prompt_list(content, job["requested_prompts"])
            usage = getattr(response, "usage", None)

            raw_handle.write(
                json.dumps(
                    {
                        "request_id": job["request_id"],
                        "dimension": job["dimension"],
                        "macro_bucket": job["macro_bucket"],
                        "requested_prompts": job["requested_prompts"],
                        "parsed_prompt_count": len(prompts),
                        "raw_content": content,
                        "usage": {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            for idx, prompt in enumerate(prompts, start=1):
                generated_handle.write(
                    json.dumps(
                        {
                            "dimension": job["dimension"],
                            "macro_bucket": job["macro_bucket"],
                            "request_id": job["request_id"],
                            "prompt_index_in_request": idx,
                            "prompt": prompt,
                            "model": model,
                            "coverage_mode": job["coverage_mode"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            total_generated += len(prompts)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    execution_summary = {
        "schema_version": "1.0",
        "model": model,
        "total_requests": total_requests,
        "total_generated_prompts": total_generated,
        "expected_target_prompts": request_plan["total_target_prompts"],
    }
    execution_summary_path.write_text(
        json.dumps(execution_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "manifest_path": str(manifest_path),
        "generated_path": str(generated_path),
        "raw_responses_path": str(raw_responses_path),
        "execution_summary_path": str(execution_summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or execute positive-prompt LLM backfill jobs.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dimensions", type=str, default=None, help="Comma-separated dimension list.")
    parser.add_argument("--max_dimensions", type=int, default=None)
    parser.add_argument("--batch_size_prompts_per_request", type=int, default=5)
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually call the configured LLM API.")
    parser.add_argument("--llm_config", type=str, default=None, help="Path to JSON/YAML LLM config.")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    args = parser.parse_args()

    dimensions = None
    if args.dimensions:
        dimensions = [item.strip() for item in args.dimensions.split(",") if item.strip()]

    request_plan = build_request_plan(
        dimensions=dimensions,
        max_dimensions=args.max_dimensions,
        batch_size_prompts_per_request=args.batch_size_prompts_per_request,
        max_requests=args.max_requests,
    )
    result = write_request_plan(
        output_dir=args.output_dir,
        dimensions=dimensions,
        max_dimensions=args.max_dimensions,
        batch_size_prompts_per_request=args.batch_size_prompts_per_request,
        max_requests=args.max_requests,
    )

    if args.execute:
        if not args.llm_config:
            raise ValueError("--llm_config is required when --execute is set")
        result = execute_request_plan(
            request_plan=request_plan,
            llm_config_path=args.llm_config,
            output_dir=args.output_dir,
            sleep_seconds=args.sleep_seconds,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
