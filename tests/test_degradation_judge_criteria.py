from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = ROOT / "data_generation" / "scripts" / "tools" / "degradation_judge.py"


def _load_judge_module():
    httpx_stub = types.ModuleType("httpx")
    httpx_stub.Client = object
    sys.modules.setdefault("httpx", httpx_stub)

    smolagents_stub = types.ModuleType("smolagents")
    smolagents_stub.tool = lambda fn: fn
    sys.modules.setdefault("smolagents", smolagents_stub)

    spec = importlib.util.spec_from_file_location("degradation_judge_under_test", JUDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DegradationJudgeCriteriaTests(unittest.TestCase):
    def test_compatibility_hints_are_loaded_from_dedicated_config(self):
        judge = _load_judge_module()

        hints = judge._load_compatibility_hints()

        self.assertIn("object_structure_error", hints)
        self.assertIn("material_mismatch", hints)
        self.assertIn("structured non-living object", hints["object_structure_error"])
        self.assertEqual(
            judge._get_dimension_criteria("object_structure_error")["compatibility_hint"],
            hints["object_structure_error"],
        )

    def test_dimension_criteria_come_from_current_sources_of_truth(self):
        judge = _load_judge_module()

        object_criteria = judge._get_dimension_criteria("object_structure_error")
        material_criteria = judge._get_dimension_criteria("material_mismatch")

        self.assertEqual(object_criteria["zh_name"], "物体结构错误")
        self.assertIn("形状扭曲", object_criteria["effect_definition"])
        self.assertIn("物体保留原类别", material_criteria["effect_definition"])
        self.assertIn("材质观感", material_criteria["effect_definition"])
        self.assertTrue(material_criteria["template_strategy_cues"])

    def test_template_strategy_cues_reflect_latest_prompt_templates(self):
        judge = _load_judge_module()

        extra_limbs = judge._get_dimension_criteria("extra_limbs")
        material = judge._get_dimension_criteria("material_mismatch")

        joined_extra = " ".join(extra_limbs["template_strategy_cues"]).lower()
        joined_material = " ".join(material["template_strategy_cues"]).lower()

        self.assertIn("extra visible arms", joined_extra)
        self.assertIn("waxy rubber", joined_material)

    def test_build_judge_prompt_includes_effect_and_compatibility_sections(self):
        judge = _load_judge_module()

        prompt = judge._build_judge_prompt(
            "extra_limbs",
            positive_prompt="a full-body adult baker in a studio",
            negative_prompt="a full-body adult baker with four visible arms in a studio",
            attribute="extra_limbs",
        )

        self.assertIn("Official effect definition:", prompt)
        self.assertIn("Template strategy cues:", prompt)
        self.assertIn("Positive compatibility hint:", prompt)
        self.assertIn("extra visible arms", prompt.lower())


if __name__ == "__main__":
    unittest.main()
