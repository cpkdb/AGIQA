from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.scripts import curator_agent_tools


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline curator agent for run-level data consolidation.")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--failure_threshold", type=int, default=2)
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_artifacts = curator_agent_tools.discover_run_artifacts(runs_root)
    failure_summary = curator_agent_tools.aggregate_failure_patterns(run_artifacts)
    failure_by_prompt = curator_agent_tools.aggregate_failure_patterns_by_prompt(run_artifacts)
    blacklist = curator_agent_tools.build_blacklist_candidates(run_artifacts, args.failure_threshold)
    decisions = curator_agent_tools.build_curation_decisions(run_artifacts, blacklist)
    memory_stats = curator_agent_tools.build_memory_stats(run_artifacts, decisions, blacklist)

    _write_json(output_dir / "failure_pattern_summary.json", failure_summary)
    _write_json(output_dir / "failure_pattern_by_prompt.json", failure_by_prompt)
    _write_json(output_dir / "blacklist.json", blacklist)
    _write_json(output_dir / "memory_stats.json", memory_stats)
    _write_jsonl(output_dir / "curation_decisions.jsonl", decisions)
    _write_json(
        output_dir / "manifest.json",
        {
            "runs_root": str(runs_root),
            "output_dir": str(output_dir),
            "failure_threshold": args.failure_threshold,
            "discovered_runs": len(run_artifacts),
        },
    )


if __name__ == "__main__":
    main()
