from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_OBJECT = ROOT / "data_generation" / "config" / "prompt_templates_v3" / "semantic_object.yaml"


class SemanticObjectTemplateTests(unittest.TestCase):
    def test_object_structure_error_templates_focus_on_large_obvious_structural_failures(self):
        source = SEMANTIC_OBJECT.read_text(encoding="utf-8")

        self.assertIn("object_structure_error:", source)
        self.assertIn("1-2 clearly visible structural defects", source)
        self.assertIn("obvious at a glance", source)
        self.assertIn("mug with a rim bent unevenly and a handle attached too high", source)
        self.assertIn("2-3 major structural defects", source)
        self.assertIn("Focus on large, visible failures in shape, part attachment, component count, or structural line geometry", source)
        self.assertIn("window frame with distorted outer edges, mismatched pane count, and corners that do not meet cleanly", source)

    def test_material_mismatch_templates_avoid_weak_sponge_like_wording(self):
        source = SEMANTIC_OBJECT.read_text(encoding="utf-8")

        self.assertIn("material_mismatch:", source)
        self.assertNotIn("sponge-like", source)
        self.assertNotIn("porous sponge", source)
        self.assertNotIn("sponge foam", source)
        self.assertNotIn("translucent jelly", source)
        self.assertNotIn("gelatinous slime", source)
        self.assertNotIn("semi-liquid", source)
        self.assertIn("cloudy plastic", source)
        self.assertIn("milky resin", source)
        self.assertIn("thick waxy rubber", source)
        self.assertIn("fabric-covered", source)

    def test_material_mismatch_templates_keep_object_identity_but_change_material(self):
        source = SEMANTIC_OBJECT.read_text(encoding="utf-8")

        self.assertIn("material_mismatch:", source)
        self.assertIn("keeps its identity and shape", source)
        self.assertIn("a wooden table that keeps its wooden shape but looks made of cloudy plastic", source)
        self.assertIn("a glass bottle that keeps its bottle shape but looks like cloudy chalky stone instead of clear hard glass", source)
        self.assertIn("a ceramic cup that keeps its shape but looks fabric-covered instead of hard and smooth", source)


if __name__ == "__main__":
    unittest.main()
