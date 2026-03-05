from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HUNYUAN_SCRIPT = ROOT / "data_generation" / "scripts" / "run_pipeline_hunyuan.sh"
SD35_SCRIPT = ROOT / "data_generation" / "scripts" / "run_pipeline_sd35_large.sh"
QWEN_SCRIPT = ROOT / "data_generation" / "scripts" / "run_pipeline_qwen_image_lightning.sh"
TRI_SMALL_BATCH_SCRIPT = ROOT / "data_generation" / "scripts" / "run_diagnostic_tri_models_small_batch.sh"


class RunScriptsTests(unittest.TestCase):
    def test_hunyuan_run_script_exists_and_targets_model(self):
        self.assertTrue(HUNYUAN_SCRIPT.exists())
        source = HUNYUAN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--model_id hunyuan-dit", source)

    def test_sd35_run_script_exists_and_targets_model(self):
        self.assertTrue(SD35_SCRIPT.exists())
        source = SD35_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--model_id sd3.5-large", source)

    def test_qwen_run_script_exists_and_targets_model(self):
        self.assertTrue(QWEN_SCRIPT.exists())
        source = QWEN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--model_id qwen-image-lightning", source)

    def test_tri_small_batch_script_exists_and_targets_three_models(self):
        self.assertTrue(TRI_SMALL_BATCH_SCRIPT.exists())
        source = TRI_SMALL_BATCH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_model flux-schnell", source)
        self.assertIn("run_model sd3.5-large", source)
        self.assertIn("run_model qwen-image-lightning", source)
        self.assertIn("--systematic", source)


if __name__ == "__main__":
    unittest.main()
