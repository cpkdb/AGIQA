from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MERGE_PATH = ROOT / "data_generation" / "scripts" / "positive_prompt_backfill_merge.py"


def load_module():
    spec = spec_from_file_location("positive_prompt_backfill_merge", MERGE_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class PositivePromptBackfillMergeTests(unittest.TestCase):
    def test_merge_script_exists(self):
        self.assertTrue(MERGE_PATH.exists())

    def test_merge_generated_backfill_rebuilds_downstream_artifacts(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_path = tmp / "working_pool_v1.jsonl"
            generated_dir = tmp / "batch_0001"
            generated_dir.mkdir(parents=True, exist_ok=True)
            generated_path = generated_dir / "generated_positive_prompts.jsonl"

            _write_jsonl(
                base_path,
                [
                    {
                        "prompt": "A ceramic mug on a wooden table",
                        "source": "diffusiondb",
                        "semantic_tags": ["has_structured_object"],
                        "signature": {
                            "tags": ["has_structured_object"],
                            "has_structured_object": True,
                        },
                        "assigned_bucket": "general_scenes",
                    }
                ],
            )
            _write_jsonl(
                generated_path,
                [
                    {
                        "dimension": "text_error",
                        "macro_bucket": "commercial_text_media",
                        "request_id": 1,
                        "prompt_index_in_request": 1,
                        "prompt": "Product packaging on a white backdrop with a centered label that reads \"ORGANIC GREEN TEA\", crisp typography and clean edges",
                        "model": "gpt-5",
                        "coverage_mode": "concentrated",
                    },
                    {
                        "dimension": "text_error",
                        "macro_bucket": "commercial_text_media",
                        "request_id": 1,
                        "prompt_index_in_request": 2,
                        "prompt": "A ceramic mug on a wooden table",
                        "model": "gpt-5",
                        "coverage_mode": "concentrated",
                    },
                ],
            )

            result = module.merge_generated_backfill(
                base_working_pool_path=str(base_path),
                generated_inputs=[str(generated_dir)],
                output_dir=str(tmp / "merged_output"),
            )

            merged_path = Path(result["merged_working_pool_path"])
            summary_path = Path(result["merge_summary_path"])
            macro_tagged_path = Path(result["macro_tagged_path"])
            coverage_path = Path(result["coverage_report_path"])
            subpool_index_path = Path(result["subpool_index_path"])
            text_subpool_path = subpool_index_path.parent / "text_error.jsonl"

            self.assertTrue(merged_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(macro_tagged_path.exists())
            self.assertTrue(coverage_path.exists())
            self.assertTrue(subpool_index_path.exists())
            self.assertTrue(text_subpool_path.exists())

            merged_records = [
                json.loads(line)
                for line in merged_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(merged_records), 2)
            llm_records = [item for item in merged_records if item["source"] == "llm_backfill"]
            self.assertEqual(len(llm_records), 1)
            self.assertIn("has_text", llm_records[0]["semantic_tags"])
            self.assertIn("has_quoted_text", llm_records[0]["semantic_tags"])
            self.assertEqual(llm_records[0]["assigned_bucket"], "llm_backfill::text_error")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["base_prompt_count"], 1)
            self.assertEqual(summary["generated_prompt_count"], 2)
            self.assertEqual(summary["deduped_new_prompt_count"], 1)
            self.assertEqual(summary["merged_prompt_count"], 2)

            text_subpool_records = [
                json.loads(line)
                for line in text_subpool_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            prompts = [item["prompt"] for item in text_subpool_records]
            self.assertIn(
                "Product packaging on a white backdrop with a centered label that reads \"ORGANIC GREEN TEA\", crisp typography and clean edges",
                prompts,
            )


if __name__ == "__main__":
    unittest.main()
