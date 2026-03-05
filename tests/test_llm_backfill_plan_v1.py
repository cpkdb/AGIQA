from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data_generation" / "data" / "llm_backfill_plan_v1.json"


class LLMBackfillPlanV1Tests(unittest.TestCase):
    def test_backfill_plan_file_exists(self):
        self.assertTrue(PLAN_PATH.exists())

    def test_backfill_plan_has_expected_priority_dimensions(self):
        data = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["source_pool"], "working_pool_v1")
        self.assertEqual(data["llm_screening_phase"], "after_merge")
        self.assertEqual(data["total_target_new_prompts"], 5200)

        dims = data["dimensions"]
        self.assertEqual([item["dimension"] for item in dims], [
            "text_error",
            "hand_malformation",
            "face_asymmetry",
            "reflection_error",
        ])

        text_plan = dims[0]
        self.assertEqual(text_plan["target_new_prompts"], 2500)
        self.assertEqual(text_plan["template_family"], "text_literal_template")
        self.assertIn("low_complexity", text_plan["complexity_mode"])

        hand_plan = dims[1]
        self.assertEqual(hand_plan["target_new_prompts"], 1200)
        self.assertEqual(hand_plan["template_family"], "hands_visible_template")

        face_plan = dims[2]
        self.assertEqual(face_plan["target_new_prompts"], 1000)
        self.assertEqual(face_plan["template_family"], "portrait_face_template")

        reflection_plan = dims[3]
        self.assertEqual(reflection_plan["target_new_prompts"], 500)
        self.assertEqual(reflection_plan["template_family"], "reflection_anchor_template")

    def test_backfill_plan_declares_generation_guardrails(self):
        data = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

        guardrails = data["global_generation_guardrails"]
        self.assertIn("forbid_quality_booster_tokens", guardrails)
        self.assertIn("length_range_words", guardrails)
        self.assertIn("complexity_should_match_dimension", guardrails)


if __name__ == "__main__":
    unittest.main()
