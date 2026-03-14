from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_PATH = ROOT / "data_generation" / "scripts" / "taxonomy_allocation_tools.py"


def _load_tools_module():
    spec = importlib.util.spec_from_file_location("taxonomy_allocation_tools", TOOLS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TaxonomyAllocationToolsTests(unittest.TestCase):
    def test_resolve_runtime_resources_prefers_latest_existing_paths(self):
        tools = _load_tools_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cleaned_source = root / "merged_working_pool_cleaned_v1.jsonl"
            cleaned_source.write_text('{"prompt": "a"}\n', encoding="utf-8")
            turbo_source = root / "merged_working_pool_sd35_turbo_clipsafe_v1.jsonl"
            turbo_source.write_text('{"prompt": "b"}\n', encoding="utf-8")

            anatomy_v2 = root / "anatomy_screened_dimension_subpools_cleaned_v2" / "index.json"
            anatomy_v2.parent.mkdir(parents=True, exist_ok=True)
            anatomy_v2.write_text(
                json.dumps({"dimensions": {"blur": {"filename": "blur.jsonl", "count": 1}}}),
                encoding="utf-8",
            )

            turbo_v2 = root / "sd35_turbo_dimension_subpools_clipsafe_v2" / "index.json"
            turbo_v2.parent.mkdir(parents=True, exist_ok=True)
            turbo_v2.write_text(
                json.dumps({"dimensions": {"blur": {"filename": "blur.jsonl", "count": 1}}}),
                encoding="utf-8",
            )

            candidates = tools.RuntimeResourceCandidates(
                base_source_prompts=root / "merged_working_pool.jsonl",
                cleaned_source_prompts=cleaned_source,
                sd35_turbo_source_prompts=turbo_source,
                base_dimension_subpool_index=root / "dimension_subpools" / "index.json",
                base_cleaned_dimension_subpool_index=root / "dimension_subpools_cleaned_v1" / "index.json",
                semantic_screened_dimension_subpool_index_v1=root / "semantic_screened_dimension_subpools_cleaned_v1" / "index.json",
                screened_cleaned_dimension_subpool_index_v2=anatomy_v2,
                screened_cleaned_dimension_subpool_index_v1=root / "anatomy_screened_dimension_subpools_cleaned_v1" / "index.json",
                screened_dimension_subpool_index=root / "anatomy_screened_dimension_subpools" / "index.json",
                sd35_turbo_semantic_screened_dimension_subpool_index_v1=root / "sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1" / "index.json",
                sd35_turbo_dimension_subpool_index_v2=turbo_v2,
                sd35_turbo_dimension_subpool_index_v1=root / "sd35_turbo_dimension_subpools_clipsafe_v1" / "index.json",
            )

            resolved = tools.resolve_runtime_resources(resource_candidates=candidates)

        self.assertEqual(
            resolved["flux-schnell"]["source_prompts"],
            str(cleaned_source),
        )
        self.assertEqual(
            resolved["qwen-image-lightning"]["dimension_subpool_index"],
            str(anatomy_v2),
        )
        self.assertEqual(
            resolved["sd3.5-large-turbo"]["source_prompts"],
            str(turbo_source),
        )
        self.assertEqual(
            resolved["sd3.5-large-turbo"]["dimension_subpool_index"],
            str(turbo_v2),
        )

    def test_inspect_pool_coverage_reads_source_and_index_counts(self):
        tools = _load_tools_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_prompts = root / "source.jsonl"
            source_prompts.write_text(
                '\n'.join(
                    [
                        json.dumps({"prompt": "p1"}),
                        json.dumps({"prompt": "p2"}),
                        json.dumps({"prompt": "p3"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            index_path = root / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "dimensions": {
                            "blur": {"filename": "blur.jsonl", "count": 3},
                            "text_error": {"filename": "text_error.jsonl", "count": 1},
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = tools.inspect_pool_coverage(
                source_prompts_path=source_prompts,
                dimension_subpool_index=index_path,
            )

        self.assertEqual(report["total_source_prompts"], 3)
        self.assertEqual(report["dimension_counts"]["blur"], 3)
        self.assertEqual(report["dimension_counts"]["text_error"], 1)
        self.assertIn("text_error", report["warnings"]["low_pool_dims"])

    def test_aggregate_run_statistics_and_speed_summary_from_artifacts(self):
        tools = _load_tools_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir)
            run_dir = runs_root / "tri_model_small_batch_20260314_120000"
            run_dir.mkdir(parents=True, exist_ok=True)

            dataset = {
                "metadata": {
                    "model_id": "flux-schnell",
                    "created_at": "2026-03-14T12:00:00",
                    "completed_at": "2026-03-14T12:01:40",
                    "total_pairs": 1,
                },
                "pairs": [{"pair_id": "pair_0000"}],
            }
            (run_dir / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")

            validation_report = {
                "summary": {
                    "total_pairs": 2,
                    "valid_pairs": 1,
                    "invalid_pairs": 1,
                    "validation_rate": 0.5,
                    "avg_attempts_per_pair": 1.5,
                    "max_retries_setting": 1,
                }
            }
            (run_dir / "validation_report.json").write_text(json.dumps(validation_report), encoding="utf-8")

            full_log = [
                {
                    "dimension": "blur",
                    "severity": "moderate",
                    "success": True,
                    "total_attempts": 1,
                    "final_validation": {"failure": None},
                },
                {
                    "dimension": "blur",
                    "severity": "severe",
                    "success": False,
                    "total_attempts": 2,
                    "final_validation": {"failure": "insufficient_effect"},
                },
            ]
            (run_dir / "full_log.json").write_text(json.dumps(full_log), encoding="utf-8")

            stats = tools.aggregate_run_statistics(runs_root=runs_root)
            speed = tools.summarize_generation_speed(runs_root=runs_root)

        self.assertEqual(stats["flux-schnell"]["blur"]["moderate"]["pairs"], 1)
        self.assertEqual(stats["flux-schnell"]["blur"]["moderate"]["valid_pairs"], 1)
        self.assertEqual(stats["flux-schnell"]["blur"]["severe"]["invalid_pairs"], 1)
        self.assertEqual(
            stats["flux-schnell"]["blur"]["severe"]["failure_types"]["insufficient_effect"],
            1,
        )
        self.assertAlmostEqual(speed["flux-schnell"]["avg_pair_seconds"], 50.0)
        self.assertAlmostEqual(speed["flux-schnell"]["avg_success_pair_seconds"], 100.0)
        self.assertAlmostEqual(speed["flux-schnell"]["avg_image_seconds"], 25.0)


if __name__ == "__main__":
    unittest.main()
