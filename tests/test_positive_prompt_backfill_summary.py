from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "data_generation" / "scripts" / "positive_prompt_backfill_summary.py"


def load_module():
    spec = spec_from_file_location("positive_prompt_backfill_summary", SUMMARY_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PositivePromptBackfillSummaryTests(unittest.TestCase):
    def test_summary_script_exists(self):
        self.assertTrue(SUMMARY_PATH.exists())

    def test_build_summary_returns_expected_top_level_counts(self):
        module = load_module()

        summary = module.build_summary()

        self.assertEqual(summary["schema_version"], "1.0")
        self.assertEqual(summary["dimension_count"], 35)
        self.assertEqual(summary["total_target_new_prompts"], 14850)
        self.assertIn("tier_totals", summary)
        self.assertEqual(
            summary["coverage_mode_totals"],
            {
                "concentrated": 4300,
                "selective_spread": 9150,
                "global_balanced": 1400,
            },
        )
        self.assertEqual(summary["tier_totals"], {"A": 9500, "B": 3950, "C": 1400})

    def test_build_summary_includes_dimension_level_allocation_details(self):
        module = load_module()

        summary = module.build_summary()
        text_error = summary["dimensions"]["text_error"]
        blur = summary["dimensions"]["blur"]

        self.assertEqual(text_error["target_new_prompts"], 2500)
        self.assertEqual(text_error["coverage_mode"], "concentrated")
        self.assertIn("commercial_text_media", text_error["macro_policy"]["core"])
        self.assertEqual(text_error["scene_mix_total"], 2500)
        self.assertEqual(text_error["scene_mix"]["commercial_text_media"], 1200)

        self.assertEqual(blur["coverage_mode"], "global_balanced")
        self.assertEqual(blur["target_new_prompts"], 100)
        self.assertEqual(blur["scene_mix_total"], 100)
        self.assertGreaterEqual(len(blur["scene_mix"]), 4)

    def test_build_priority_queue_sorts_by_priority_then_target(self):
        module = load_module()

        queue = module.build_priority_queue(limit=5)

        self.assertEqual(len(queue), 5)
        self.assertEqual(queue[0]["dimension"], "text_error")
        self.assertEqual(queue[0]["priority"], 1)
        self.assertGreaterEqual(queue[0]["target_new_prompts"], queue[1]["target_new_prompts"])


if __name__ == "__main__":
    unittest.main()
