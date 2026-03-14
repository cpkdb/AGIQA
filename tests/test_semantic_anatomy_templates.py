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
        self.assertIn("too many legs, one missing leg, two heads on one neck", source)
        self.assertIn("an extra tail, a wing growing from the wrong place", source)
        self.assertIn("Avoid subtle direction-only mistakes", source)
        self.assertIn("Combine 2-3 strong, obvious anatomy failures", source)
        self.assertIn("The anatomy failure should be obvious at first glance", source)

    def test_body_proportion_error_templates_cover_multiple_proportion_failure_families(self):
        source = SEMANTIC_ANATOMY.read_text(encoding="utf-8")

        self.assertIn("body_proportion_error:", source)
        self.assertIn("head scale or leg length", source)
        self.assertIn("head that looks noticeably too small for the body", source)
        self.assertIn("unusually long legs beneath a compact torso", source)
        self.assertIn("oversized head + very short legs", source)
        self.assertIn("tiny head + very long legs", source)
        self.assertIn("TWO dominant proportion errors that are easy to read at a glance", source)
        self.assertIn("easy to read at a glance", source)

    def test_hand_malformation_templates_cover_multiple_hand_failure_families(self):
        source = SEMANTIC_ANATOMY.read_text(encoding="utf-8")

        self.assertIn("hand_malformation:", source)
        self.assertIn("one extra finger", source)
        self.assertIn("partially fused fingers", source)
        self.assertIn("uneven finger lengths or thickness", source)
        self.assertIn("awkwardly placed thumb", source)

    def test_extra_limbs_templates_prefer_extra_arms_without_overconstraining_visibility_wording(self):
        source = SEMANTIC_ANATOMY.read_text(encoding="utf-8")

        self.assertIn("extra_limbs:", source)
        self.assertIn("Strongly prefer extra visible arms", source)
        self.assertIn("Prefer four visible arms", source)
        self.assertIn("Prefer severe extra-arm", source)
        self.assertIn("Prefer six visible arms", source)
        self.assertNotIn("distinct hands", source)
        self.assertNotIn("fully visible", source)
        self.assertNotIn("spatially separated", source)
        self.assertNotIn("all fully visible in the frame", source)


if __name__ == "__main__":
    unittest.main()
