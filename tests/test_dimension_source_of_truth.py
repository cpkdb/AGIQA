from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "data_generation" / "docs" / "degradation_dimensions.md"
QUALITY_PATH = ROOT / "data_generation" / "config" / "quality_dimensions_active.json"
LEGACY_QUALITY_PATH = ROOT / "data_generation" / "config" / "quality_dimensions_v3.json"
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


class DimensionSourceOfTruthTests(unittest.TestCase):
    def test_quality_dimensions_file_has_been_renamed_to_active(self):
        self.assertTrue(QUALITY_PATH.exists())
        self.assertFalse(LEGACY_QUALITY_PATH.exists())

    def test_quality_dimensions_match_doc_dimension_rows(self):
        doc_dims = parse_doc_dimensions()
        quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))

        quality_dims = []
        for perspective in quality["perspectives"].values():
            quality_dims.extend(perspective["dimensions"].keys())

        self.assertEqual(doc_dims, quality_dims)
        self.assertEqual(len(doc_dims), 35)

    def test_renderer_active_dimensions_match_doc_dimension_rows(self):
        doc_dims = parse_doc_dimensions()
        renderer = load_renderer()

        active_dims = [item["dimension"] for item in renderer.list_active_dimension_plans()]

        self.assertEqual(active_dims, doc_dims)
        self.assertIn("awkward_positioning", active_dims)
        self.assertIn("awkward_framing", active_dims)
        self.assertNotIn("unexpected_element", active_dims)
        self.assertNotIn("amateur_look", active_dims)


if __name__ == "__main__":
    unittest.main()
