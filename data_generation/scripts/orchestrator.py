from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.scripts import orchestrator_tools


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight orchestrator for normalized run planning.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", required=True, choices=sorted(orchestrator_tools.DEFAULT_MODEL_RUN_SETTINGS))
    parser.add_argument("--subcategory_filter")
    parser.add_argument("--attribute_filter")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = orchestrator_tools.build_run_id(args.model_id)
    run_config = orchestrator_tools.build_run_config(
        model_id=args.model_id,
        run_id=run_id,
        output_root=orchestrator_tools.DEFAULT_OUTPUT_ROOT,
        subcategory_filter=args.subcategory_filter,
        attribute_filter=args.attribute_filter,
    )
    run_config_path = output_dir / "run_config.json"
    registry_path = output_dir / "run_registry.json"
    launch_path = output_dir / "launch_command.sh"

    _write_json(run_config_path, run_config)
    _write_json(registry_path, orchestrator_tools.build_run_registry(run_config, run_config_path))
    launch_path.write_text("#!/bin/bash\nset -e\n" + orchestrator_tools.build_launch_command(run_config) + "\n", encoding="utf-8")
    launch_path.chmod(0o755)

    if args.execute and not args.dry_run:
        log_path = output_dir / "launch.log"
        orchestrator_tools.update_run_registry(
            registry_path,
            status="running",
            extra={"started_at": orchestrator_tools.utc_now_iso(), "log_path": str(log_path.resolve())},
        )
        return_code = orchestrator_tools.execute_launch_script(launch_path, log_path)
        orchestrator_tools.update_run_registry(
            registry_path,
            status="completed" if return_code == 0 else "failed",
            extra={"completed_at": orchestrator_tools.utc_now_iso(), "return_code": return_code},
        )
        if return_code != 0:
            raise SystemExit(return_code)


if __name__ == "__main__":
    main()
