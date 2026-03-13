from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
QWEN_GENERATOR = ROOT / "data_generation" / "scripts" / "qwen_image_lightning_generator.py"
IMAGE_GENERATOR = ROOT / "data_generation" / "scripts" / "tools" / "image_generator.py"
PIPELINE = ROOT / "data_generation" / "scripts" / "pipeline.py"
QWEN_RUN_SCRIPT = ROOT / "data_generation" / "scripts" / "run_pipeline_qwen_image_lightning.sh"
TRI_DIAG_SCRIPT = ROOT / "data_generation" / "scripts" / "run_diagnostic_tri_models_small_batch.sh"
DIAG_SCRIPT = ROOT / "data_generation" / "scripts" / "run_diagnostic.sh"
OFFICIAL_QWEN_SNAPSHOT = "/root/autodl-tmp/AGIQA/Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"
NUNCHAKU_INT4_4STEP = "nunchaku-ai/nunchaku-qwen-image/svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
LOCAL_NUNCHAKU_INT4_4STEP = "/root/autodl-tmp/AGIQA/Nunchaku/svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"


class QwenRuntimeDefaultsTests(unittest.TestCase):
    def test_qwen_registration_defaults_to_fit_24g_without_nf4(self):
        source = IMAGE_GENERATOR.read_text(encoding="utf-8")

        self.assertIn('use_nf4: bool = False', source)
        self.assertIn('runtime_profile: str = "fit-24g"', source)

    def test_qwen_generator_matches_official_scheduler_and_scoped_nf4(self):
        source = QWEN_GENERATOR.read_text(encoding="utf-8")

        self.assertIn(f'DEFAULT_BASE_MODEL_ID = "{OFFICIAL_QWEN_SNAPSHOT}"', source)
        self.assertIn('"shift": 1.0', source)
        self.assertIn('"use_dynamic_shifting": True', source)
        self.assertIn('components_to_quantize=["transformer", "text_encoder"]', source)
        self.assertIn('runtime_profile == "fast-gpu-24g"', source)
        self.assertIn('enable_model_cpu_offload', source)
        self.assertIn('fast-gpu-24g 在当前 GPU 上 OOM，自动回退到 offload', source)
        self.assertIn('self.pipe = self.pipe.to(device)', source)

    def test_qwen_generator_exposes_nunchaku_int4_profile(self):
        source = QWEN_GENERATOR.read_text(encoding="utf-8")

        self.assertIn('runtime_profile == "nunchaku-int4"', source)
        self.assertIn(f'DEFAULT_NUNCHAKU_MODEL = "{NUNCHAKU_INT4_4STEP}"', source)
        self.assertIn('NunchakuQwenImageTransformer2DModel', source)
        self.assertIn('QwenImagePipeline', source)
        self.assertIn('self.pipe._exclude_from_cpu_offload.append("transformer")', source)
        self.assertNotIn('self.pipe.load_lora_weights(self.lora_repo, weight_name=self.weight_name)', source.split('runtime_profile == "nunchaku-int4"')[-1])

    def test_qwen_run_script_defaults_to_fit_24g_profile(self):
        source = QWEN_RUN_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(f'MODEL_PATH="{OFFICIAL_QWEN_SNAPSHOT}"', source)
        self.assertIn('RUNTIME_PROFILE="fast-gpu-24g"', source)
        self.assertIn('USE_CPU_OFFLOAD=false', source)
        self.assertIn(f'NUNCHAKU_MODEL_PATH="${{NUNCHAKU_MODEL_PATH:-{LOCAL_NUNCHAKU_INT4_4STEP}}}"', source)
        self.assertIn('if [ "$RUNTIME_PROFILE" = "nunchaku-int4" ]; then', source)
        self.assertIn('--nunchaku_model_path $NUNCHAKU_MODEL_PATH', source)

    def test_qwen_triage_script_uses_fit_24g_profile(self):
        source = TRI_DIAG_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(f'QWEN_IMAGE_LIGHTNING_MODEL_PATH="${{QWEN_IMAGE_LIGHTNING_MODEL_PATH:-{OFFICIAL_QWEN_SNAPSHOT}}}"', source)
        self.assertIn('run_model qwen-image-lightning "$QWEN_IMAGE_LIGHTNING_MODEL_PATH" 4 1.0 fast-gpu-24g', source)
        self.assertIn(f'QWEN_IMAGE_LIGHTNING_NUNCHAKU_MODEL_PATH="${{QWEN_IMAGE_LIGHTNING_NUNCHAKU_MODEL_PATH:-{LOCAL_NUNCHAKU_INT4_4STEP}}}"', source)
        self.assertIn('if [[ "$runtime_profile" == "nunchaku-int4" ]]; then', source)
        self.assertIn('--nunchaku_model_path "$QWEN_IMAGE_LIGHTNING_NUNCHAKU_MODEL_PATH"', source)

    def test_qwen_diagnostic_script_uses_official_snapshot_path(self):
        source = DIAG_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(f'QWEN_IMAGE_LIGHTNING_MODEL_PATH="{OFFICIAL_QWEN_SNAPSHOT}"', source)

    def test_pipeline_rejects_empty_negative_prompt_before_generation(self):
        source = PIPELINE.read_text(encoding="utf-8")

        self.assertIn('negative_prompt = degrade_result["negative_prompt"]', source)
        self.assertIn('if not negative_prompt or not negative_prompt.strip():', source)
        self.assertIn('attempt_record["error"] = "empty_negative_prompt"', source)


if __name__ == "__main__":
    unittest.main()
