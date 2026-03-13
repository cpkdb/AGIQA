from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_ANATOMY = ROOT / "data_generation" / "config" / "prompt_templates_v3" / "semantic_anatomy.yaml"


class SemanticAnatomyTemplateTests(unittest.TestCase):
    def test_face_asymmetry_templates_removed_from_active_anatomy_templates(self):
        source = SEMANTIC_ANATOMY.read_text(encoding="utf-8")

        self.assertNotIn("face_asymmetry:", source)

    def test_animal_anatomy_error_templates_cover_multiple_anatomy_failure_families(self):
        source = SEMANTIC_ANATOMY.read_text(encoding="utf-8")

        self.assertIn("animal_anatomy_error:", source)
        self.assertIn("moderate: |", source)
        self.assertIn("severe: |", source)
        self.assertIn("limb structure or count", source)
        self.assertIn("joint direction", source)
        self.assertIn("head, neck, or body connection", source)
        self.assertIn("body proportion or silhouette", source)
        self.assertIn("different kinds of structural errors", source)

    def test_body_proportion_error_templates_cover_multiple_proportion_failure_families(self):
        source = SEMANTIC_ANATOMY.read_text(encoding="utf-8")

        self.assertIn("body_proportion_error:", source)
        self.assertIn("oversized head + very short legs", source)
        self.assertIn("tiny head + oversized upper body", source)
        self.assertIn("broad shoulders with a bulky torso", source)
        self.assertIn("TWO dominant proportion errors that are easy to read at a glance", source)
        self.assertIn("easy to read at a glance", source)

    def test_hand_malformation_templates_cover_multiple_hand_failure_families(self):
        source = SEMANTIC_ANATOMY.read_text(encoding="utf-8")

        self.assertIn("hand_malformation:", source)
        self.assertIn("one extra finger", source)
        self.assertIn("partially fused fingers", source)
        self.assertIn("uneven finger lengths or thickness", source)
        self.assertIn("awkwardly placed thumb", source)


if __name__ == "__main__":
    unittest.main()
