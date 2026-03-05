#!/usr/bin/env python3
"""
Build and optionally execute small resumable batches for positive prompt backfill.

This wrapper prevents oversized single runs by splitting the full request plan into
many small request batches. Each batch gets its own subdirectory and request plan.
"""

from __future__ import annotations

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
EXECUTOR_PATH = SCRIPT_DIR / "positive_prompt_backfill_executor.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "positive_prompt_backfill_batch_runs"


def _load_executor_module():
    spec = spec_from_file_location("positive_prompt_backfill_executor", EXECUTOR_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _chunk_jobs(jobs: List[Dict], requests_per_batch: int) -> List[List[Dict]]:
    if requests_per_batch <= 0:
        raise ValueError("requests_per_batch must be positive")
    return [jobs[i : i + requests_per_batch] for i in range(0, len(jobs), requests_per_batch)]


def build_batched_execution_plan(
    dimensions: Optional[List[str]] = None,
    max_dimensions: Optional[int] = None,
    batch_size_prompts_per_request: int = 5,
    max_requests: Optional[int] = None,
    requests_per_batch: int = 10,
) -> Dict:
    executor = _load_executor_module()
    request_plan = executor.build_request_plan(
        dimensions=dimensions,
        max_dimensions=max_dimensions,
        batch_size_prompts_per_request=batch_size_prompts_per_request,
        max_requests=max_requests,
    )

    batches: List[Dict] = []
    for batch_index, batch_jobs in enumerate(_chunk_jobs(request_plan["jobs"], requests_per_batch), start=1):
        batch_dimensions = sorted({job["dimension"] for job in batch_jobs})
        batches.append(
            {
                "batch_index": batch_index,
                "request_count": len(batch_jobs),
                "request_ids": [job["request_id"] for job in batch_jobs],
                "dimensions": batch_dimensions,
                "request_plan": {
                    "schema_version": request_plan["schema_version"],
                    "dimension_count": len(batch_dimensions),
                    "selected_dimensions": batch_dimensions,
                    "total_target_prompts": sum(job["requested_prompts"] for job in batch_jobs),
                    "batch_size_prompts_per_request": request_plan["batch_size_prompts_per_request"],
                    "total_requests": len(batch_jobs),
                    "jobs": batch_jobs,
                },
            }
        )

    return {
        "schema_version": "1.0",
        "total_requests": request_plan["total_requests"],
        "batch_count": len(batches),
        "batch_size_prompts_per_request": batch_size_prompts_per_request,
        "requests_per_batch": requests_per_batch,
        "selected_dimensions": request_plan["selected_dimensions"],
        "batches": batches,
    }


def write_batched_execution_plan(
    output_dir: str,
    dimensions: Optional[List[str]] = None,
    max_dimensions: Optional[int] = None,
    batch_size_prompts_per_request: int = 5,
    max_requests: Optional[int] = None,
    requests_per_batch: int = 10,
) -> Dict[str, str]:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batches_root = output_root / "batches"
    batches_root.mkdir(parents=True, exist_ok=True)

    batch_plan = build_batched_execution_plan(
        dimensions=dimensions,
        max_dimensions=max_dimensions,
        batch_size_prompts_per_request=batch_size_prompts_per_request,
        max_requests=max_requests,
        requests_per_batch=requests_per_batch,
    )

    batch_index_payload = {
        key: value
        for key, value in batch_plan.items()
        if key != "batches"
    }
    batch_index_payload["batches"] = []

    for batch in batch_plan["batches"]:
        batch_dir = batches_root / f"batch_{batch['batch_index']:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = batch_dir / "request_plan.json"
        manifest_path.write_text(
            json.dumps(batch["request_plan"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        batch_index_payload["batches"].append(
            {
                "batch_index": batch["batch_index"],
                "request_count": batch["request_count"],
                "request_ids": batch["request_ids"],
                "dimensions": batch["dimensions"],
                "manifest_path": str(manifest_path),
                "batch_dir": str(batch_dir),
            }
        )

    batch_index_path = output_root / "batch_index.json"
    batch_index_path.write_text(
        json.dumps(batch_index_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"batch_index_path": str(batch_index_path)}


def ensure_batched_execution_plan(
    output_dir: str,
    dimensions: Optional[List[str]] = None,
    max_dimensions: Optional[int] = None,
    batch_size_prompts_per_request: int = 5,
    max_requests: Optional[int] = None,
    requests_per_batch: int = 10,
) -> Dict[str, str]:
    output_root = Path(output_dir).resolve()
    batch_index_path = output_root / "batch_index.json"
    if batch_index_path.exists():
        return {"batch_index_path": str(batch_index_path)}
    return write_batched_execution_plan(
        output_dir=output_dir,
        dimensions=dimensions,
        max_dimensions=max_dimensions,
        batch_size_prompts_per_request=batch_size_prompts_per_request,
        max_requests=max_requests,
        requests_per_batch=requests_per_batch,
    )


def get_batch_status(output_dir: str) -> Dict:
    output_root = Path(output_dir).resolve()
    batch_index_path = output_root / "batch_index.json"
    if not batch_index_path.exists():
        raise FileNotFoundError(f"Batch index not found: {batch_index_path}")

    payload = json.loads(batch_index_path.read_text(encoding="utf-8"))
    status_batches: List[Dict] = []
    completed_batches = 0
    next_pending_batch_index = None

    for batch in payload["batches"]:
        batch_dir = Path(batch["batch_dir"])
        generated_exists = (batch_dir / "generated_positive_prompts.jsonl").exists()
        summary_exists = (batch_dir / "execution_summary.json").exists()
        status = "completed" if generated_exists and summary_exists else "pending"
        if status == "completed":
            completed_batches += 1
        elif next_pending_batch_index is None:
            next_pending_batch_index = batch["batch_index"]

        status_batches.append(
            {
                **batch,
                "status": status,
                "generated_exists": generated_exists,
                "execution_summary_exists": summary_exists,
            }
        )

    return {
        "schema_version": "1.0",
        "batch_count": payload["batch_count"],
        "completed_batches": completed_batches,
        "pending_batches": payload["batch_count"] - completed_batches,
        "next_pending_batch_index": next_pending_batch_index,
        "batches": status_batches,
    }


def execute_batch(
    output_dir: str,
    batch_index: int,
    llm_config_path: str,
    sleep_seconds: float = 0.0,
) -> Dict[str, str]:
    output_root = Path(output_dir).resolve()
    batch_index_path = output_root / "batch_index.json"
    if not batch_index_path.exists():
        raise FileNotFoundError(f"Batch index not found: {batch_index_path}")

    payload = json.loads(batch_index_path.read_text(encoding="utf-8"))
    target = None
    for batch in payload["batches"]:
        if batch["batch_index"] == batch_index:
            target = batch
            break
    if target is None:
        raise KeyError(f"Unknown batch index: {batch_index}")

    manifest_path = Path(target["manifest_path"])
    request_plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    executor = _load_executor_module()
    return executor.execute_request_plan(
        request_plan=request_plan,
        llm_config_path=llm_config_path,
        output_dir=target["batch_dir"],
        sleep_seconds=sleep_seconds,
    )


def execute_next_pending_batch(
    output_dir: str,
    llm_config_path: str,
    sleep_seconds: float = 0.0,
) -> Dict[str, str]:
    status = get_batch_status(output_dir)
    batch_index = status["next_pending_batch_index"]
    if batch_index is None:
        raise ValueError("No pending batches remain")

    result = execute_batch(
        output_dir=output_dir,
        batch_index=batch_index,
        llm_config_path=llm_config_path,
        sleep_seconds=sleep_seconds,
    )
    result["executed_batch_index"] = batch_index
    return result


def execute_all_pending_batches(
    output_dir: str,
    llm_config_path: str,
    sleep_seconds: float = 0.0,
) -> Dict[str, object]:
    executed_batch_indices: List[int] = []
    last_result: Optional[Dict[str, str]] = None

    while True:
        status = get_batch_status(output_dir)
        next_batch = status["next_pending_batch_index"]
        if next_batch is None:
            break
        result = execute_next_pending_batch(
            output_dir=output_dir,
            llm_config_path=llm_config_path,
            sleep_seconds=sleep_seconds,
        )
        executed_batch_indices.append(result["executed_batch_index"])
        last_result = result

    final_status = get_batch_status(output_dir)
    summary: Dict[str, object] = {
        "executed_batch_count": len(executed_batch_indices),
        "executed_batch_indices": executed_batch_indices,
        "completed_batches": final_status["completed_batches"],
        "pending_batches": final_status["pending_batches"],
        "next_pending_batch_index": final_status["next_pending_batch_index"],
    }
    if last_result is not None:
        summary["last_batch_result"] = last_result
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or execute batched positive-prompt backfill jobs.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dimensions", type=str, default=None, help="Comma-separated dimension list.")
    parser.add_argument("--max_dimensions", type=int, default=None)
    parser.add_argument("--batch_size_prompts_per_request", type=int, default=5)
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--requests_per_batch", type=int, default=10)
    parser.add_argument("--execute_batch", type=int, default=None, help="Execute one prepared batch index.")
    parser.add_argument("--execute_next_pending", action="store_true", help="Execute the next pending batch.")
    parser.add_argument("--execute_all_pending", action="store_true", help="Execute all pending batches sequentially.")
    parser.add_argument("--status", action="store_true", help="Show current batch completion status.")
    parser.add_argument("--llm_config", type=str, default=None)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    args = parser.parse_args()

    dimensions = None
    if args.dimensions:
        dimensions = [item.strip() for item in args.dimensions.split(",") if item.strip()]

    result = ensure_batched_execution_plan(
        output_dir=args.output_dir,
        dimensions=dimensions,
        max_dimensions=args.max_dimensions,
        batch_size_prompts_per_request=args.batch_size_prompts_per_request,
        max_requests=args.max_requests,
        requests_per_batch=args.requests_per_batch,
    )

    if args.status:
        result = get_batch_status(args.output_dir)

    if args.execute_batch is not None:
        if not args.llm_config:
            raise ValueError("--llm_config is required when --execute_batch is set")
        result = execute_batch(
            output_dir=args.output_dir,
            batch_index=args.execute_batch,
            llm_config_path=args.llm_config,
            sleep_seconds=args.sleep_seconds,
        )
    elif args.execute_next_pending:
        if not args.llm_config:
            raise ValueError("--llm_config is required when --execute_next_pending is set")
        result = execute_next_pending_batch(
            output_dir=args.output_dir,
            llm_config_path=args.llm_config,
            sleep_seconds=args.sleep_seconds,
        )
    elif args.execute_all_pending:
        if not args.llm_config:
            raise ValueError("--llm_config is required when --execute_all_pending is set")
        result = execute_all_pending_batches(
            output_dir=args.output_dir,
            llm_config_path=args.llm_config,
            sleep_seconds=args.sleep_seconds,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
