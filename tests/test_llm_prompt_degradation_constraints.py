from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LLM_DEGRADER = ROOT / "data_generation" / "scripts" / "llm_prompt_degradation.py"


def load_module():
    spec = spec_from_file_location("llm_prompt_degradation_under_test", LLM_DEGRADER)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptDegradationConstraintTests(unittest.TestCase):
    def setUp(self):
        module = load_module()
        self.cls = module.LLMPromptDegradation

    def _build_degrader(self):
        degrader = self.cls.__new__(self.cls)
        degrader.system_prompt_cache = {
            "technical_quality_blur_moderate": "BASE SYSTEM PROMPT"
        }
        degrader.subcategory_descriptions = {
            "technical_quality": {"category": "technical_quality"}
        }
        degrader.config = {"degradation": {"validate_output": False}}
        degrader._build_system_prompt = lambda subcategory, severity: "FALLBACK"
        return degrader

    def test_sd35_turbo_appends_77_token_constraint(self):
        degrader = self._build_degrader()
        captured = {}

        def fake_call(system_prompt, user_prompt):
            captured["system_prompt"] = system_prompt
            return "negative prompt"

        degrader._call_llm_api = fake_call

        degrader.generate_negative_prompt(
            positive_prompt="a portrait of a woman in dramatic light",
            subcategory="technical_quality",
            attribute="blur",
            severity="moderate",
            model_id="sd3.5-large-turbo",
        )

        self.assertIn("BASE SYSTEM PROMPT", captured["system_prompt"])
        self.assertIn("within 77 CLIP tokens", captured["system_prompt"])

    def test_non_sd35_models_do_not_get_turbo_only_constraint(self):
        degrader = self._build_degrader()
        captured = {}

        def fake_call(system_prompt, user_prompt):
            captured["system_prompt"] = system_prompt
            return "negative prompt"

        degrader._call_llm_api = fake_call

        degrader.generate_negative_prompt(
            positive_prompt="a portrait of a woman in dramatic light",
            subcategory="technical_quality",
            attribute="blur",
            severity="moderate",
            model_id="flux-schnell",
        )

        self.assertEqual(captured["system_prompt"], "BASE SYSTEM PROMPT")

    def test_validate_and_fix_output_strips_markdown_asterisks(self):
        degrader = self._build_degrader()
        degrader.config = {"degradation": {"min_length": 1, "max_length": 200}}

        cleaned = degrader._validate_and_fix_output(
            'people watching **a rocket as large as an entire mountain range** launch',
            "people watching a rocket launch",
        )

        self.assertEqual(
            cleaned,
            "people watching a rocket as large as an entire mountain range launch",
        )


if __name__ == "__main__":
    unittest.main()
