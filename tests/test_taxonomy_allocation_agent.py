from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = ROOT / "data_generation" / "scripts" / "taxonomy_allocation_agent.py"


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("taxonomy_allocation_agent", AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TaxonomyAllocationAgentTests(unittest.TestCase):
    def test_agent_writes_expected_analysis_artifacts(self):
        agent = _load_agent_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs_root = root / "runs"
            runs_root.mkdir()
            output_dir = root / "agent_output"

            run_dir = runs_root / "run_001"
            run_dir.mkdir()
            (run_dir / "dataset.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "model_id": "qwen-image-lightning",
                            "created_at": "2026-03-14T12:00:00",
                            "completed_at": "2026-03-14T12:00:20",
                            "total_pairs": 1,
                        },
                        "pairs": [{"pair_id": "pair_0000"}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "validation_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "total_pairs": 1,
                            "valid_pairs": 1,
                            "invalid_pairs": 0,
                            "validation_rate": 1.0,
                            "avg_attempts_per_pair": 1.0,
                            "max_retries_setting": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "full_log.json").write_text(
                json.dumps(
                    [
                        {
                            "dimension": "blur",
                            "severity": "moderate",
                            "success": True,
                            "total_attempts": 1,
                            "final_validation": {"failure": None},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            source_prompts = root / "merged_working_pool_cleaned_v1.jsonl"
            source_prompts.write_text(json.dumps({"prompt": "p1"}) + "\n", encoding="utf-8")
            index_path = root / "anatomy_screened_dimension_subpools_cleaned_v2" / "index.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                json.dumps({"dimensions": {"blur": {"filename": "blur.jsonl", "count": 1}}}),
                encoding="utf-8",
            )
            turbo_source = root / "merged_working_pool_sd35_turbo_clipsafe_v1.jsonl"
            turbo_source.write_text(json.dumps({"prompt": "p2"}) + "\n", encoding="utf-8")
            turbo_index = root / "sd35_turbo_dimension_subpools_clipsafe_v2" / "index.json"
            turbo_index.parent.mkdir(parents=True, exist_ok=True)
            turbo_index.write_text(
                json.dumps({"dimensions": {"blur": {"filename": "blur.jsonl", "count": 1}}}),
                encoding="utf-8",
            )

            taxonomy_path = root / "quality_dimensions_active.json"
            taxonomy_path.write_text(
                json.dumps(
                    {
                        "taxonomy_name": "test_taxonomy",
                        "statistics": {"total_dimensions": 2},
                        "perspectives": {
                            "technical_quality": {
                                "dimensions": {
                                    "blur": {"zh": "模糊"},
                                }
                            },
                            "semantic_rationality": {
                                "dimensions": {
                                    "extra_limbs": {"zh": "额外肢体"},
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            candidates = agent.tools.RuntimeResourceCandidates(
                base_source_prompts=root / "merged_working_pool.jsonl",
                cleaned_source_prompts=source_prompts,
                sd35_turbo_source_prompts=turbo_source,
                base_dimension_subpool_index=root / "dimension_subpools" / "index.json",
                base_cleaned_dimension_subpool_index=root / "dimension_subpools_cleaned_v1" / "index.json",
                semantic_screened_dimension_subpool_index_v1=root / "semantic_screened_dimension_subpools_cleaned_v1" / "index.json",
                screened_cleaned_dimension_subpool_index_v2=index_path,
                screened_cleaned_dimension_subpool_index_v1=root / "anatomy_screened_dimension_subpools_cleaned_v1" / "index.json",
                screened_dimension_subpool_index=root / "anatomy_screened_dimension_subpools" / "index.json",
                sd35_turbo_semantic_screened_dimension_subpool_index_v1=root / "sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1" / "index.json",
                sd35_turbo_dimension_subpool_index_v2=turbo_index,
                sd35_turbo_dimension_subpool_index_v1=root / "sd35_turbo_dimension_subpools_clipsafe_v1" / "index.json",
            )

            result = agent.run_taxonomy_allocation_agent(
                output_dir=output_dir,
                runs_root=runs_root,
                taxonomy_path=taxonomy_path,
                resource_candidates=candidates,
            )

            expected_files = {
                "coverage_summary": output_dir / "coverage_summary.json",
                "model_dimension_stats": output_dir / "model_dimension_stats.json",
                "speed_summary": output_dir / "speed_summary.json",
                "allocation_plan_template": output_dir / "allocation_plan.template.json",
                "allocation_insights": output_dir / "allocation_insights.md",
                "manifest": output_dir / "manifest.json",
            }

            for file_path in expected_files.values():
                self.assertTrue(file_path.exists(), str(file_path))

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"], {key: str(path) for key, path in expected_files.items()})
            self.assertEqual(result["artifacts"], manifest["artifacts"])

            insights = (output_dir / "allocation_insights.md").read_text(encoding="utf-8")
            self.assertIn("# Taxonomy & Allocation Insights", insights)
            self.assertIn("## Runtime Resources", insights)
            self.assertIn("## Pool Coverage", insights)
            self.assertIn("## Historical Success", insights)
            self.assertIn("## Speed Summary", insights)


if __name__ == "__main__":
    unittest.main()
