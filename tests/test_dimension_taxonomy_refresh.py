from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "data_generation" / "docs" / "degradation_dimensions.md"
ACTIVE_TAXONOMY_PATH = ROOT / "data_generation" / "config" / "quality_dimensions_active.json"


class DimensionTaxonomyRefreshTests(unittest.TestCase):
    def test_main_doc_removes_unstable_dimensions_and_reworks_object_integrity(self):
        source = DOC_PATH.read_text(encoding="utf-8")

        self.assertNotIn("**face_asymmetry**", source)
        self.assertNotIn("**impossible_pose**", source)
        self.assertNotIn("**shadow_mismatch**", source)
        self.assertNotIn("**reflection_error**", source)
        self.assertNotIn("**part_attachment_error**", source)
        self.assertNotIn("**transparency_opacity_error**", source)

        self.assertIn("**object_structure_error**", source)
        self.assertIn("**material_mismatch**", source)

    def test_active_taxonomy_removes_face_attachment_and_transparency_and_renames_object_shape(self):
        payload = json.loads(ACTIVE_TAXONOMY_PATH.read_text(encoding="utf-8"))
        semantic_dims = payload["perspectives"]["semantic_rationality"]["dimensions"]

        self.assertNotIn("face_asymmetry", semantic_dims)
        self.assertNotIn("impossible_pose", semantic_dims)
        self.assertNotIn("shadow_mismatch", semantic_dims)
        self.assertNotIn("reflection_error", semantic_dims)
        self.assertNotIn("part_attachment_error", semantic_dims)
        self.assertNotIn("transparency_opacity_error", semantic_dims)
        self.assertNotIn("object_shape_error", semantic_dims)

        self.assertIn("object_structure_error", semantic_dims)
        self.assertIn("material_mismatch", semantic_dims)

        self.assertEqual(payload["statistics"]["total_dimensions"], 32)


if __name__ == "__main__":
    unittest.main()
