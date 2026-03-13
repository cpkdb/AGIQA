from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HUNYUAN_SCRIPT = ROOT / "data_generation" / "scripts" / "run_pipeline_hunyuan.sh"
SD35_SCRIPT = ROOT / "data_generation" / "scripts" / "run_pipeline_sd35_large.sh"
QWEN_SCRIPT = ROOT / "data_generation" / "scripts" / "run_pipeline_qwen_image_lightning.sh"
TRI_SMALL_BATCH_SCRIPT = ROOT / "data_generation" / "scripts" / "run_diagnostic_tri_models_small_batch.sh"
RUN_PIPELINE_SCRIPT = ROOT / "data_generation" / "scripts" / "run_pipeline.sh"


class RunScriptsTests(unittest.TestCase):
    def test_hunyuan_run_script_exists_and_targets_model(self):
        self.assertTrue(HUNYUAN_SCRIPT.exists())
        source = HUNYUAN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--model_id hunyuan-dit", source)

    def test_sd35_run_script_exists_and_targets_model(self):
        self.assertTrue(SD35_SCRIPT.exists())
        source = SD35_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--model_id sd3.5-large-turbo", source)
        self.assertIn("merged_working_pool_sd35_turbo_clipsafe_v1.jsonl", source)
        self.assertIn("sd35_turbo_dimension_subpools_clipsafe_v2/index.json", source)
        self.assertIn("sd35_turbo_dimension_subpools_clipsafe_v1/index.json", source)
        self.assertNotIn("sd35_turbo_targeted_dimension_subpools_clipsafe_v1/index.json", source)

    def test_qwen_run_script_exists_and_targets_model(self):
        self.assertTrue(QWEN_SCRIPT.exists())
        source = QWEN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--model_id qwen-image-lightning", source)
        self.assertIn("--systematic", source)
        self.assertIn("--dimension_subpool_index $DIMENSION_SUBPOOL_INDEX", source)
        self.assertIn("merged_working_pool_cleaned_v1.jsonl", source)
        self.assertIn("dimension_subpools_cleaned_v1/index.json", source)
        self.assertIn("anatomy_screened_dimension_subpools_cleaned_v2/index.json", source)
        self.assertIn("anatomy_screened_dimension_subpools_cleaned_v1/index.json", source)
        self.assertNotIn("targeted_dimension_subpools_cleaned_v1/index.json", source)
        self.assertIn("export JUDGE_CONFIG_PATH=", source)
        self.assertIn("judge_config_api_gpt_ge.yaml", source)

    def test_tri_small_batch_script_exists_and_targets_three_models(self):
        self.assertTrue(TRI_SMALL_BATCH_SCRIPT.exists())
        source = TRI_SMALL_BATCH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_model flux-schnell", source)
        self.assertIn("run_model sd3.5-large", source)
        self.assertIn("run_model qwen-image-lightning", source)
        self.assertIn("--systematic", source)
        self.assertIn("--dimension_subpool_index \"$dimension_subpool_index\"", source)
        self.assertIn("merged_working_pool_cleaned_v1.jsonl", source)
        self.assertIn("merged_working_pool_sd35_turbo_clipsafe_v1.jsonl", source)
        self.assertIn("dimension_subpools_cleaned_v1/index.json", source)
        self.assertIn("anatomy_screened_dimension_subpools_cleaned_v2/index.json", source)
        self.assertIn("anatomy_screened_dimension_subpools_cleaned_v1/index.json", source)
        self.assertIn("sd35_turbo_dimension_subpools_clipsafe_v2/index.json", source)
        self.assertIn("sd35_turbo_dimension_subpools_clipsafe_v1/index.json", source)
        self.assertNotIn("targeted_dimension_subpools_cleaned_v1/index.json", source)
        self.assertNotIn("sd35_turbo_targeted_dimension_subpools_clipsafe_v1/index.json", source)
        self.assertIn("unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy", source)
        self.assertIn("export JUDGE_CONFIG_PATH=", source)
        self.assertIn("judge_config_api_gpt_ge.yaml", source)
        self.assertIn("run_model qwen-image-lightning \"$QWEN_IMAGE_LIGHTNING_MODEL_PATH\" 4 1.0 nunchaku-int4", source)

    def test_generic_run_script_can_switch_to_turbo_specific_pool(self):
        source = RUN_PIPELINE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("merged_working_pool_sd35_turbo_clipsafe_v1.jsonl", source)
        self.assertIn("anatomy_screened_dimension_subpools_cleaned_v2/index.json", source)
        self.assertIn("sd35_turbo_dimension_subpools_clipsafe_v2/index.json", source)
        self.assertIn("sd35_turbo_dimension_subpools_clipsafe_v1/index.json", source)
        self.assertNotIn("targeted_dimension_subpools_cleaned_v1/index.json", source)
        self.assertNotIn("sd35_turbo_targeted_dimension_subpools_clipsafe_v1/index.json", source)


if __name__ == "__main__":
    unittest.main()
