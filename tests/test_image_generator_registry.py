from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMAGE_GENERATOR_PATH = ROOT / "data_generation" / "scripts" / "tools" / "image_generator.py"


class ImageGeneratorRegistryTests(unittest.TestCase):
    def test_image_generator_supports_hunyuan_dit_model_id(self):
        source = IMAGE_GENERATOR_PATH.read_text(encoding="utf-8")

        self.assertIn('"hunyuan-dit"', source)

    def test_image_generator_supports_sd35_large_model_id(self):
        source = IMAGE_GENERATOR_PATH.read_text(encoding="utf-8")

        self.assertIn('"sd3.5-large"', source)

    def test_image_generator_supports_qwen_image_lightning_model_id(self):
        source = IMAGE_GENERATOR_PATH.read_text(encoding="utf-8")

        self.assertIn('"qwen-image-lightning"', source)


if __name__ == "__main__":
    unittest.main()
