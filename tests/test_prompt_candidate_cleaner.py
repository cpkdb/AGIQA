from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLEANER_PATH = ROOT / "data_generation" / "scripts" / "prompt_candidate_cleaner.py"


def load_module():
    spec = spec_from_file_location("prompt_candidate_cleaner", CLEANER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptCandidateCleanerTests(unittest.TestCase):
    def test_prompt_candidate_cleaner_script_exists(self):
        self.assertTrue(CLEANER_PATH.exists())

    def test_normalize_prompt_collapses_whitespace(self):
        module = load_module()

        normalized = module.normalize_prompt("  a   cat\n\n on\t the mat  ")

        self.assertEqual(normalized, "a cat on the mat")

    def test_clean_and_dedup_candidates_filters_short_and_exact_duplicates(self):
        module = load_module()

        records = [
            {"source": "diffusiondb", "prompt": "  a   cat  on the mat  "},
            {"source": "pick_a_pic_v2", "prompt": "a cat on the mat"},
            {"source": "parti_prompts", "prompt": "ok"},
            {"source": "longbench_t2i", "prompt": ""},
            {"source": "t2i_compbench", "prompt": "a dog in the park"},
        ]

        cleaned, summary = module.clean_and_dedup_candidates(records, min_words=3)

        self.assertEqual([item["prompt"] for item in cleaned], [
            "a cat on the mat",
            "a dog in the park",
        ])
        self.assertEqual(summary["input_count"], 5)
        self.assertEqual(summary["filtered_empty_count"], 1)
        self.assertEqual(summary["filtered_short_count"], 1)
        self.assertEqual(summary["exact_duplicate_count"], 1)
        self.assertEqual(summary["output_count"], 2)
        self.assertEqual(summary["source_counts"]["diffusiondb"], 1)
        self.assertEqual(summary["source_counts"]["t2i_compbench"], 1)

    def test_t2i_compbench_allows_shorter_structured_prompts(self):
        module = load_module()

        records = [
            {"source": "t2i_compbench", "prompt": "two apples"},
            {"source": "diffusiondb", "prompt": "two apples"},
        ]

        cleaned, summary = module.clean_and_dedup_candidates(records, min_words=3)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["source"], "t2i_compbench")
        self.assertEqual(cleaned[0]["prompt"], "two apples")
        self.assertEqual(summary["filtered_short_count"], 1)
        self.assertEqual(summary["output_count"], 1)

    def test_clean_candidate_file_writes_outputs(self):
        module = load_module()

        records = [
            {"source": "diffusiondb", "record_id": 1, "prompt": "a red apple on a wooden table"},
            {"source": "parti_prompts", "record_id": 2, "prompt": "a red apple on a wooden table"},
            {"source": "longbench_t2i", "record_id": 3, "prompt": "three birds flying over a lake"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in records
                ),
                encoding="utf-8",
            )

            outputs = module.clean_candidate_file(
                input_path=str(input_path),
                output_dir=str(tmpdir_path),
                min_words=3,
            )

            cleaned_path = Path(outputs["cleaned_path"])
            summary_path = Path(outputs["summary_path"])
            self.assertTrue(cleaned_path.exists())
            self.assertTrue(summary_path.exists())

            cleaned_lines = cleaned_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(cleaned_lines), 2)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["input_count"], 3)
            self.assertEqual(summary["exact_duplicate_count"], 1)
            self.assertEqual(summary["output_count"], 2)


if __name__ == "__main__":
    unittest.main()
