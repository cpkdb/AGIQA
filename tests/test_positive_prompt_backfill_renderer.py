from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = ROOT / "data_generation" / "scripts" / "positive_prompt_backfill_renderer.py"


def load_module():
    spec = spec_from_file_location("positive_prompt_backfill_renderer", RENDERER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PositivePromptBackfillRendererTests(unittest.TestCase):
    def test_renderer_script_exists(self):
        self.assertTrue(RENDERER_PATH.exists())

    def test_get_dimension_plan_returns_expected_dimension_entry(self):
        module = load_module()

        plan = module.get_dimension_plan("text_error")

        self.assertEqual(plan["dimension"], "text_error")
        self.assertEqual(plan["coverage_mode"], "concentrated")
        self.assertEqual(plan["template_family"], "text_literal_template")
        self.assertIn("commercial_text_media", plan["macro_taxonomy_policy_override"]["core"])

    def test_get_dimension_plan_returns_doc_backfilled_dimension_defaults(self):
        module = load_module()

        plan = module.get_dimension_plan("awkward_positioning")

        self.assertEqual(plan["dimension"], "awkward_positioning")
        self.assertEqual(plan["tier"], "C")
        self.assertEqual(plan["target_new_prompts"], 100)
        self.assertEqual(plan["template_family"], "generic_quality_anchor_template")

    def test_render_generation_prompt_includes_core_and_optional_macro_buckets(self):
        module = load_module()

        rendered = module.render_generation_prompt("text_error")

        self.assertEqual(rendered["dimension"], "text_error")
        self.assertIn("coverage_mode", rendered)
        self.assertEqual(rendered["coverage_mode"], "concentrated")
        self.assertIn("commercial_text_media", rendered["system_prompt"])
        self.assertIn("objects_products_tools", rendered["system_prompt"])
        self.assertIn("core macro buckets", rendered["system_prompt"])
        self.assertIn("optional macro buckets", rendered["system_prompt"])
        self.assertIn("Output only", rendered["system_prompt"])
        self.assertNotIn("Target dimension support: text_error.", rendered["system_prompt"])
        self.assertIn("without mentioning internal degradation labels", rendered["system_prompt"])
        self.assertEqual(sum(rendered["scene_mix"].values()), 2500)

    def test_render_generation_prompt_for_global_balanced_dimension_uses_template_defaults(self):
        module = load_module()

        rendered = module.render_generation_prompt("blur")

        self.assertEqual(rendered["dimension"], "blur")
        self.assertEqual(rendered["coverage_mode"], "global_balanced")
        self.assertNotIn("scene_mix_override", rendered["dimension_plan"])
        self.assertGreater(len(rendered["scene_mix"]), 3)
        self.assertEqual(sum(rendered["scene_mix"].values()), 100)
        self.assertIn("global macro balance", rendered["system_prompt"])

    def test_render_generation_prompt_uses_default_coverage_mode_when_not_explicit(self):
        module = load_module()

        rendered = module.render_generation_prompt("expression_mismatch")

        self.assertEqual(rendered["dimension"], "expression_mismatch")
        self.assertEqual(rendered["coverage_mode"], "selective_spread")
        self.assertEqual(sum(rendered["scene_mix"].values()), 800)
        self.assertIn("Coverage mode: selective_spread.", rendered["system_prompt"])

    def test_render_job_prompt_localizes_to_single_macro_bucket(self):
        module = load_module()

        rendered = module.render_job_prompt(
            dimension="text_error",
            macro_bucket="commercial_text_media",
            requested_prompts=5,
        )

        self.assertEqual(rendered["dimension"], "text_error")
        self.assertEqual(rendered["macro_bucket"], "commercial_text_media")
        self.assertEqual(rendered["scene_mix"], {"commercial_text_media": 5})
        self.assertIn("This request is only for the macro bucket: commercial_text_media.", rendered["system_prompt"])
        self.assertNotIn("Target dimension support: text_error.", rendered["system_prompt"])
        self.assertNotIn("- objects_products_tools: 500", rendered["system_prompt"])

    def test_render_batch_plan_returns_multiple_dimensions(self):
        module = load_module()

        batch = module.render_batch_plan(["text_error", "hand_malformation"])

        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0]["dimension"], "text_error")
        self.assertEqual(batch[1]["dimension"], "hand_malformation")

    def test_list_active_dimension_plans_uses_doc_source_of_truth(self):
        module = load_module()

        active = module.list_active_dimension_plans()
        active_dimensions = [item["dimension"] for item in active]

        self.assertEqual(len(active_dimensions), 35)
        self.assertIn("awkward_framing", active_dimensions)
        self.assertNotIn("unexpected_element", active_dimensions)
        self.assertNotIn("amateur_look", active_dimensions)


if __name__ == "__main__":
    unittest.main()
