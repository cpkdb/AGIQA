from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_TEXT = ROOT / "data_generation" / "config" / "prompt_templates_v3" / "semantic_text.yaml"


class SemanticTextTemplateTests(unittest.TestCase):
    def test_text_error_templates_cover_more_than_spelling_only(self):
        source = SEMANTIC_TEXT.read_text(encoding="utf-8")

        self.assertIn("text_error:", source)
        self.assertIn("spacing or segmentation", source)
        self.assertIn("partially unreadable", source)
        self.assertIn("mixed readable and unreadable fragments", source)

    def test_logo_symbol_error_templates_cover_multiple_intrusion_types(self):
        source = SEMANTIC_TEXT.read_text(encoding="utf-8")

        self.assertIn("small attached sticker or label", source)
        self.assertIn("barcode, QR-like mark, inspection tag, price label", source)
        self.assertIn("small printed sign or instruction card", source)
        self.assertIn("repeated small symbol contamination", source)


if __name__ == "__main__":
    unittest.main()
