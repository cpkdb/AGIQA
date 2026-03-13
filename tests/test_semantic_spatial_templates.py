from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SPATIAL = ROOT / "data_generation" / "config" / "prompt_templates_v3" / "semantic_spatial.yaml"


class SemanticSpatialTemplateTests(unittest.TestCase):
    def test_scale_inconsistency_templates_focus_on_local_scene_internal_size_mismatch(self):
        source = SEMANTIC_SPATIAL.read_text(encoding="utf-8")

        self.assertIn("scale_inconsistency:", source)
        self.assertIn("nearby reference objects", source)
        self.assertIn("local scale error rather than a fantasy giant-world transformation", source)
        self.assertIn("an insect larger than the fruit beside it", source)
        self.assertIn("without turning the whole image into a surreal giant-world scene", source)

    def test_floating_objects_templates_extend_beyond_zero_gravity_only(self):
        source = SEMANTIC_SPATIAL.read_text(encoding="utf-8")

        self.assertIn("floating_objects:", source)
        self.assertIn("visible gap", source)
        self.assertIn("without support beneath", source)
        self.assertIn("hovering above a table, floor, or ground", source)
        self.assertIn("multiple objects or the main subject", source)
        self.assertIn("without changing the scene into a fantasy zero-gravity world", source)

    def test_penetration_overlap_templates_cover_overlap_merge_and_attachment_failures(self):
        source = SEMANTIC_SPATIAL.read_text(encoding="utf-8")

        self.assertIn("penetration_overlap:", source)
        self.assertIn("impossible overlap or merge", source)
        self.assertIn("biting into each other", source)
        self.assertIn("impossible overlap, merge, or attachment", source)
        self.assertIn("attached in an impossible way", source)

    def test_transparency_opacity_error_templates_removed_from_active_spatial_templates(self):
        source = SEMANTIC_SPATIAL.read_text(encoding="utf-8")

        self.assertNotIn("transparency_opacity_error:", source)

    def test_context_mismatch_templates_stay_on_context_not_neighbor_dimensions(self):
        source = SEMANTIC_SPATIAL.read_text(encoding="utf-8")

        self.assertIn("context_mismatch:", source)
        self.assertIn("era, habitat, or scene purpose", source)
        self.assertIn("not from physics, placement, or time-of-day", source)
        self.assertIn("wrong setting or context decision", source)
        self.assertIn("not as layout error, temporal conflict, or extra-object clutter", source)


if __name__ == "__main__":
    unittest.main()
