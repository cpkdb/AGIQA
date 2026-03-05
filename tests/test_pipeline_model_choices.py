from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "data_generation" / "scripts" / "pipeline.py"
DEMO_PATH = ROOT / "data_generation" / "scripts" / "demo_v3_dimension_paired.py"


class PipelineModelChoicesTests(unittest.TestCase):
    def test_pipeline_exposes_new_model_ids(self):
        source = PIPELINE_PATH.read_text(encoding="utf-8")

        self.assertIn('"hunyuan-dit"', source)
        self.assertIn('"sd3.5-large"', source)
        self.assertIn('"qwen-image-lightning"', source)

    def test_demo_script_exposes_new_model_ids(self):
        source = DEMO_PATH.read_text(encoding="utf-8")

        self.assertIn('"hunyuan-dit"', source)
        self.assertIn('"sd3.5-large"', source)
        self.assertIn('"qwen-image-lightning"', source)


if __name__ == "__main__":
    unittest.main()
