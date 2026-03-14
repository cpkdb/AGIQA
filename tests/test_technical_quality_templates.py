from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TECHNICAL_QUALITY = ROOT / "data_generation" / "config" / "prompt_templates_v3" / "technical_quality.yaml"


class TechnicalQualityTemplateTests(unittest.TestCase):
    def test_blur_templates_push_global_focus_failure_for_portraits(self):
        source = TECHNICAL_QUALITY.read_text(encoding="utf-8")

        self.assertIn("blur:", source)
        self.assertIn("make the blur affect the overall readability of the image", source)
        self.assertIn("the eyes, eyelashes, eyebrows, hair strands, skin detail, and clothing edges should all read as softly blurred", source)
        self.assertIn("overall soft unreadability", source)
        self.assertIn("eyes, lashes, brows, lips, hair strands, and clothing edges should all blur together", source)


if __name__ == "__main__":
    unittest.main()
