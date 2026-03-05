from pathlib import Path
import json
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "data_generation" / "docs" / "degradation_dimensions.md"
SEMANTIC_PATH = ROOT / "data_generation" / "config" / "semantic_tag_requirements.json"


def parse_doc_dimensions():
    text = DOC_PATH.read_text(encoding="utf-8")
    return [
        match.group(1)
        for match in re.finditer(r"^\| \*\*([a-z0-9_]+)\*\* \|", text, flags=re.MULTILINE)
    ]


class SemanticTagRequirementsSyncTests(unittest.TestCase):
    def test_dimension_requirements_match_doc_dimensions(self):
        doc_dimensions = parse_doc_dimensions()
        config = json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))

        semantic_dimensions = []
        for _, dims in config["dimension_requirements"].items():
            if isinstance(dims, dict):
                semantic_dimensions.extend([k for k in dims.keys() if not k.startswith("_")])

        self.assertEqual(semantic_dimensions, doc_dimensions)
        self.assertEqual(config["summary"]["total_dimensions"], 35)
        self.assertNotIn("unexpected_element", semantic_dimensions)
        self.assertIn("awkward_positioning", semantic_dimensions)
        self.assertIn("awkward_framing", semantic_dimensions)


if __name__ == "__main__":
    unittest.main()
