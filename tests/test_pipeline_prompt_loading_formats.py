from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "data_generation" / "scripts" / "pipeline.py"


class PipelinePromptLoadingFormatTests(unittest.TestCase):
    def test_pipeline_supports_jsonl_loading(self):
        source = PIPELINE_PATH.read_text(encoding="utf-8")
        self.assertIn('suffix == ".jsonl"', source)
        self.assertIn("Invalid JSONL", source)

    def test_pipeline_uses_signature_alias_and_safe_model_filter(self):
        source = PIPELINE_PATH.read_text(encoding="utf-8")
        self.assertIn('item.get("prompt_signature", item.get("signature"))', source)
        self.assertIn("if model_filter and model_name and model_filter not in model_name", source)


if __name__ == "__main__":
    unittest.main()
