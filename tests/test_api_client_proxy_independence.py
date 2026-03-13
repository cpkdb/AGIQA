from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os
import tempfile
import unittest
import yaml


ROOT = Path(__file__).resolve().parents[1]
LLM_DEGRADER = ROOT / "data_generation" / "scripts" / "llm_prompt_degradation.py"
JUDGE = ROOT / "data_generation" / "scripts" / "tools" / "degradation_judge.py"
PROMPT_DEGRADER = ROOT / "data_generation" / "scripts" / "tools" / "prompt_degrader.py"


def load_judge_module():
    spec = spec_from_file_location("degradation_judge_under_test", JUDGE)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ApiClientProxyIndependenceTests(unittest.TestCase):
    def test_prompt_degrader_defaults_to_alternate_gpt5_provider_config(self):
        source = PROMPT_DEGRADER.read_text(encoding="utf-8")

        self.assertIn('llm_config_api_gpt_ge.yaml', source)

    def test_llm_degrader_uses_httpx_client_with_trust_env_disabled(self):
        source = LLM_DEGRADER.read_text(encoding="utf-8")

        self.assertIn("import httpx", source)
        self.assertIn("trust_env=False", source)
        self.assertIn("httpx.Client(", source)

    def test_judge_uses_httpx_client_with_trust_env_disabled(self):
        source = JUDGE.read_text(encoding="utf-8")

        self.assertIn("import httpx", source)
        self.assertIn("trust_env=False", source)
        self.assertIn("http_client=httpx.Client(", source)

    def test_judge_supports_config_path_override_via_env(self):
        module = load_judge_module()
        original = os.environ.get("JUDGE_CONFIG_PATH")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "judge_override.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "vlm": {
                            "api_key": "dummy",
                            "api_base": "https://example.com/v1",
                            "model": "test-model",
                            "timeout": 10,
                        },
                        "judge": {"strict_mode": False},
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            os.environ["JUDGE_CONFIG_PATH"] = str(config_path)
            module._config = None
            loaded = module._load_config()

            self.assertEqual(loaded["vlm"]["api_base"], "https://example.com/v1")
            self.assertEqual(loaded["vlm"]["model"], "test-model")

        module._config = None
        if original is None:
            os.environ.pop("JUDGE_CONFIG_PATH", None)
        else:
            os.environ["JUDGE_CONFIG_PATH"] = original


if __name__ == "__main__":
    unittest.main()
