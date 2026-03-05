from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import json
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data_generation" / "data" / "llm_backfill_plan_all_dimensions_v1.json"
COST_PATH = ROOT / "data_generation" / "data" / "llm_backfill_cost_estimate_v1.json"
TAXONOMY_PATH = ROOT / "data_generation" / "data" / "prompt_macro_taxonomy_v1.json"
DOC_PATH = ROOT / "data_generation" / "docs" / "degradation_dimensions.md"
RENDERER_PATH = ROOT / "data_generation" / "scripts" / "positive_prompt_backfill_renderer.py"


def load_renderer():
    spec = spec_from_file_location("positive_prompt_backfill_renderer", RENDERER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_doc_dimensions():
    text = DOC_PATH.read_text(encoding="utf-8")
    return [
        match.group(1)
        for match in re.finditer(r"^\| \*\*([a-z0-9_]+)\*\* \|", text, flags=re.MULTILINE)
    ]


class LLMBackfillPlanAllDimensionsV1Tests(unittest.TestCase):
    def test_all_dimensions_backfill_plan_exists(self):
        self.assertTrue(PLAN_PATH.exists())

    def test_all_dimensions_backfill_plan_covers_full_dimension_set(self):
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        renderer = load_renderer()
        expected_dimensions = parse_doc_dimensions()
        planned_dimensions = [item["dimension"] for item in plan["dimensions"]]
        active_dimensions = [item["dimension"] for item in renderer.list_active_dimension_plans()]
        active_total = sum(
            item["target_new_prompts"] for item in renderer.list_active_dimension_plans()
        )

        self.assertEqual(plan["schema_version"], "1.0")
        self.assertEqual(plan["macro_taxonomy_ref"], "prompt_macro_taxonomy_v1")
        self.assertEqual(plan["macro_taxonomy_strategy"], "macro_scene_plus_micro_constraints")
        self.assertIn("coverage_modes", plan)
        self.assertIn("default_coverage_mode_by_tier", plan)
        self.assertIn("default_macro_taxonomy_policy", plan)
        self.assertIn("default_scene_mix_policy_by_template_family", plan)
        self.assertEqual(planned_dimensions, expected_dimensions)
        self.assertEqual(len(planned_dimensions), 35)
        self.assertEqual(plan["total_target_new_prompts"], 14850)
        self.assertEqual(len(active_dimensions), 35)
        self.assertEqual(active_dimensions, expected_dimensions)
        self.assertEqual(len(set(active_dimensions)), 35)
        self.assertEqual(active_total, 14850)

    def test_all_dimensions_backfill_plan_has_expected_template_mappings(self):
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        index = {item["dimension"]: item for item in plan["dimensions"]}

        self.assertEqual(index["text_error"]["template_family"], "text_literal_template")
        self.assertEqual(index["text_error"]["target_new_prompts"], 2500)
        self.assertEqual(index["hand_malformation"]["template_family"], "hands_visible_template")
        self.assertEqual(index["face_asymmetry"]["template_family"], "portrait_face_template")
        self.assertEqual(index["count_error"]["template_family"], "relation_count_template")
        self.assertEqual(index["scene_layout_error"]["template_family"], "multi_object_layout_template")
        self.assertEqual(index["blur"]["template_family"], "generic_quality_anchor_template")
        self.assertEqual(index["logo_symbol_error"]["template_family"], "text_literal_template")
        self.assertEqual(index["text_error"]["coverage_mode"], "concentrated")
        self.assertEqual(index["hand_malformation"]["coverage_mode"], "selective_spread")
        self.assertEqual(index["face_asymmetry"]["coverage_mode"], "concentrated")
        self.assertEqual(index["reflection_error"]["coverage_mode"], "selective_spread")
        self.assertEqual(index["count_error"]["coverage_mode"], "selective_spread")
        self.assertEqual(index["scene_layout_error"]["coverage_mode"], "selective_spread")
        self.assertEqual(index["blur"]["coverage_mode"], "global_balanced")
        self.assertIn("macro_taxonomy_policy_override", index["text_error"])
        self.assertIn("commercial_text_media", index["text_error"]["macro_taxonomy_policy_override"]["core"])
        self.assertIn("human_activity_interaction", index["hand_malformation"]["macro_taxonomy_policy_override"]["core"])
        self.assertIn("human_portrait", index["face_asymmetry"]["macro_taxonomy_policy_override"]["core"])
        self.assertIn("natural_landscape", index["reflection_error"]["macro_taxonomy_policy_override"]["core"])
        self.assertEqual(sum(index["text_error"]["scene_mix_override"].values()), index["text_error"]["target_new_prompts"])
        self.assertEqual(sum(index["hand_malformation"]["scene_mix_override"].values()), index["hand_malformation"]["target_new_prompts"])
        self.assertEqual(sum(index["face_asymmetry"]["scene_mix_override"].values()), index["face_asymmetry"]["target_new_prompts"])
        self.assertEqual(sum(index["reflection_error"]["scene_mix_override"].values()), index["reflection_error"]["target_new_prompts"])
        self.assertEqual(sum(index["count_error"]["scene_mix_override"].values()), index["count_error"]["target_new_prompts"])
        self.assertEqual(sum(index["scene_layout_error"]["scene_mix_override"].values()), index["scene_layout_error"]["target_new_prompts"])
        self.assertEqual(sum(index["logo_symbol_error"]["scene_mix_override"].values()), index["logo_symbol_error"]["target_new_prompts"])
        self.assertNotIn("scene_mix_override", index["blur"])

    def test_macro_taxonomy_file_exists_and_has_expected_categories(self):
        self.assertTrue(TAXONOMY_PATH.exists())

        taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
        categories = taxonomy["categories"]

        self.assertEqual(taxonomy["schema_version"], "1.0")
        self.assertEqual(len(categories), 10)
        self.assertEqual(
            set(categories.keys()),
            {
                "human_portrait",
                "human_activity_interaction",
                "indoor_structured_space",
                "urban_built_outdoor",
                "natural_landscape",
                "animals_creatures",
                "objects_products_tools",
                "food_tabletop",
                "commercial_text_media",
                "events_performance_culture",
            },
        )

    def test_cost_estimate_file_exists_and_matches_plan(self):
        self.assertTrue(COST_PATH.exists())

        cost = json.loads(COST_PATH.read_text(encoding="utf-8"))
        renderer = load_renderer()
        active_total = sum(
            item["target_new_prompts"] for item in renderer.list_active_dimension_plans()
        )

        self.assertEqual(cost["schema_version"], "1.0")
        self.assertEqual(cost["model"], "gpt-4o")
        self.assertEqual(cost["assumptions"]["target_new_prompts"], active_total)
        self.assertEqual(cost["assumptions"]["batch_size_prompts_per_request"], 5)
        self.assertAlmostEqual(cost["estimated_requests"], 2970)
        self.assertGreater(cost["estimated_generation_cost_usd"], 10.0)
        self.assertLess(cost["estimated_generation_cost_usd"], 15.0)
        self.assertGreater(cost["estimated_generation_plus_screening_cost_usd"], cost["estimated_generation_cost_usd"])


if __name__ == "__main__":
    unittest.main()
