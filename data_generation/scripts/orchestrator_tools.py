from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from data_generation.scripts import prompt_pool_agent_tools


DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/orchestrated_runs")
PIPELINE_PATH = Path("/root/ImageReward/data_generation/scripts/pipeline.py")


DEFAULT_MODEL_RUN_SETTINGS: Dict[str, Dict[str, Any]] = {
    "flux-schnell": {
        "model_path": None,
        "runtime_profile": "fast-gpu",
        "steps": 4,
        "cfg": 0.0,
        "model_filter": "sdxl",
    },
    "qwen-image-lightning": {
        "model_path": "/root/autodl-tmp/AGIQA/Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6",
        "runtime_profile": "fast-gpu-24g",
        "steps": 4,
        "cfg": 1.0,
        "model_filter": None,
    },
    "sd3.5-large-turbo": {
        "model_path": "/root/autodl-tmp/AGIQA/sd3.5-large-turbo",
        "runtime_profile": "fit-24g",
        "steps": 4,
        "cfg": 1.0,
        "model_filter": None,
    },
    "sdxl": {
        "model_path": "/root/ckpts/sd_xl_base_1.0.safetensors",
        "runtime_profile": "fast-gpu",
        "steps": 35,
        "cfg": 7.5,
        "model_filter": "sdxl",
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id(model_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{model_id}_{timestamp}"


def resolve_active_resources(model_id: str) -> Dict[str, str]:
    resources = prompt_pool_agent_tools.resolve_active_prompt_pools(model_ids=[model_id])
    selected = resources.get(model_id) or {}
    return {
        "source_prompts": selected.get("source_prompts") or "",
        "dimension_subpool_index": selected.get("dimension_subpool_index") or "",
    }


def build_run_config(
    *,
    model_id: str,
    run_id: str,
    output_root: Path,
    subcategory_filter: str | None = None,
    attribute_filter: str | None = None,
) -> Dict[str, Any]:
    model_settings = DEFAULT_MODEL_RUN_SETTINGS[model_id]
    resources = resolve_active_resources(model_id)
    output_dir = (output_root / run_id / model_id).resolve()
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "model_id": model_id,
        "model_path": model_settings["model_path"],
        "runtime_profile": model_settings["runtime_profile"],
        "steps": model_settings["steps"],
        "cfg": model_settings["cfg"],
        "model_filter": model_settings["model_filter"],
        "source_prompts": resources["source_prompts"],
        "dimension_subpool_index": resources["dimension_subpool_index"],
        "output_dir": str(output_dir),
        "num_pairs_per_prompt": 3,
        "max_retries": 2,
        "seed": 42,
        "severities": "moderate,severe",
        "shuffle": True,
        "systematic": True,
        "subcategory_filter": subcategory_filter,
        "attribute_filter": attribute_filter,
        "pipeline_path": str(PIPELINE_PATH),
    }


def build_launch_command(run_config: Dict[str, Any]) -> str:
    model_path_value = json.dumps(run_config["model_path"]) if run_config["model_path"] else '""'
    python_parts = [
        "python scripts/pipeline.py",
        f"--source_prompts {json.dumps(run_config['source_prompts'])}",
        f"--output_dir {json.dumps(run_config['output_dir'])}",
        f"--model_id {run_config['model_id']}",
        f"--model_path {model_path_value}",
        f"--runtime_profile {run_config['runtime_profile']}",
        f"--num_pairs_per_prompt {run_config['num_pairs_per_prompt']}",
        f"--max_retries {run_config['max_retries']}",
        f"--seed {run_config['seed']}",
        f"--severities {run_config['severities']}",
        f"--steps {run_config['steps']}",
        f"--cfg {run_config['cfg']}",
        "--shuffle",
        "--systematic",
        f"--dimension_subpool_index {json.dumps(run_config['dimension_subpool_index'])}",
    ]
    if run_config.get("model_filter"):
        python_parts.append(f"--model_filter {run_config['model_filter']}")
    if run_config.get("subcategory_filter"):
        python_parts.append(f"--subcategory_filter {run_config['subcategory_filter']}")
    if run_config.get("attribute_filter"):
        python_parts.append(f"--attribute_filter {run_config['attribute_filter']}")
    python_command = " \\\n  ".join(python_parts)
    return "\n".join(
        [
            "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy",
            "cd /root/ImageReward/data_generation",
            python_command,
        ]
    )


def build_run_registry(run_config: Dict[str, Any], run_config_path: Path) -> Dict[str, Any]:
    return {
        "run_id": run_config["run_id"],
        "created_at": run_config["created_at"],
        "status": "planned",
        "model_id": run_config["model_id"],
        "source_prompts": run_config["source_prompts"],
        "dimension_subpool_index": run_config["dimension_subpool_index"],
        "run_config_path": str(run_config_path.resolve()),
    }


def update_run_registry(registry_path: Path, *, status: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["status"] = status
    if extra:
        payload.update(extra)
    registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def execute_launch_script(launch_path: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            ["bash", str(launch_path)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return int(completed.returncode)
