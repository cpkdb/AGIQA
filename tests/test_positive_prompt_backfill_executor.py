from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PATH = ROOT / "data_generation" / "scripts" / "positive_prompt_backfill_executor.py"


def load_module():
    spec = spec_from_file_location("positive_prompt_backfill_executor", EXECUTOR_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PositivePromptBackfillExecutorTests(unittest.TestCase):
    def test_executor_script_exists(self):
        self.assertTrue(EXECUTOR_PATH.exists())

    def test_build_request_plan_for_single_dimension_splits_scene_mix_into_jobs(self):
        module = load_module()

        plan = module.build_request_plan(
            dimensions=["text_error"],
            batch_size_prompts_per_request=1000,
        )

        self.assertEqual(plan["dimension_count"], 1)
        self.assertEqual(plan["total_target_prompts"], 2500)
        self.assertEqual(plan["total_requests"], 6)
        self.assertEqual(plan["jobs"][0]["dimension"], "text_error")
        self.assertEqual(plan["jobs"][0]["macro_bucket"], "commercial_text_media")
        self.assertEqual(plan["jobs"][0]["requested_prompts"], 1000)
        self.assertIn("only for the macro bucket: commercial_text_media", plan["jobs"][0]["system_prompt"])
        self.assertIn("explicit text carrier", plan["jobs"][0]["user_prompt"])
        self.assertIn("quoted readable literal text", plan["jobs"][0]["user_prompt"])
        self.assertIn("must not already exhibit the target degradation", plan["jobs"][0]["user_prompt"])
        self.assertIn("text itself must be correct and clearly readable", plan["jobs"][0]["user_prompt"])
        self.assertNotIn("degradation dimension 'text_error'", plan["jobs"][0]["user_prompt"])
        self.assertIn("support the assigned downstream contrastive check", plan["jobs"][0]["user_prompt"])

    def test_build_request_plan_uses_priority_queue_when_dimensions_not_provided(self):
        module = load_module()

        plan = module.build_request_plan(
            max_dimensions=1,
            batch_size_prompts_per_request=2000,
        )

        self.assertEqual(plan["dimension_count"], 1)
        self.assertEqual({job["dimension"] for job in plan["jobs"]}, {"text_error"})

    def test_write_request_plan_writes_manifest_and_jobs_json(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = module.write_request_plan(
                output_dir=tmpdir,
                dimensions=["face_asymmetry"],
                batch_size_prompts_per_request=500,
            )

            manifest_path = Path(result["manifest_path"])
            self.assertTrue(manifest_path.exists())

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dimension_count"], 1)
            self.assertEqual(payload["total_target_prompts"], 1000)
            self.assertEqual(payload["total_requests"], 5)

    def test_load_llm_config_supports_yaml_without_pyyaml(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "llm:",
                        '  provider: "openai"',
                        '  model: "gpt-4o-mini"',
                        '  api_key: "test-key"',
                        '  api_base: "https://example.com"',
                        "  temperature: 0.3",
                        "  max_tokens: 256",
                    ]
                ),
                encoding="utf-8",
            )

            real_import = __import__

            def fake_import(name, *args, **kwargs):
                if name == "yaml":
                    raise ImportError("forced missing yaml")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                config = module._load_llm_config(str(config_path))

            self.assertEqual(config["llm"]["provider"], "openai")
            self.assertEqual(config["llm"]["model"], "gpt-4o-mini")
            self.assertEqual(config["llm"]["api_key"], "test-key")
            self.assertEqual(config["llm"]["api_base"], "https://example.com")
            self.assertEqual(config["llm"]["temperature"], 0.3)
            self.assertEqual(config["llm"]["max_tokens"], 256)

    def test_parse_prompt_list_extracts_json_array_from_markdown_fence(self):
        module = load_module()

        raw = """Here is the JSON array:

```json
["prompt a", "prompt b", "prompt c"]
```"""

        prompts = module._parse_prompt_list(raw, expected_count=2)

        self.assertEqual(prompts, ["prompt a", "prompt b"])

    def test_execute_request_plan_writes_raw_responses_and_parses_fenced_json(self):
        module = load_module()

        request_plan = {
            "schema_version": "1.0",
            "dimension_count": 1,
            "selected_dimensions": ["text_error"],
            "total_target_prompts": 2,
            "batch_size_prompts_per_request": 2,
            "total_requests": 1,
            "jobs": [
                {
                    "request_id": 1,
                    "dimension": "text_error",
                    "priority": 1,
                    "tier": "Tier A",
                    "coverage_mode": "concentrated",
                    "macro_bucket": "commercial_text_media",
                    "requested_prompts": 2,
                    "chunk_index": 1,
                    "template_family": "text_literal_template",
                    "system_prompt": "system",
                    "user_prompt": "user",
                }
            ],
        }

        response_text = """Sure.
```json
["alpha", "beta"]
```"""

        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: fake_response,
                )
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm_config.json"
            config_path.write_text(
                json.dumps({"llm": {"api_key": "test-key", "model": "gpt-4o-mini"}}),
                encoding="utf-8",
            )

            with patch.object(module, "_create_openai_client", return_value=fake_client):
                result = module.execute_request_plan(
                    request_plan=request_plan,
                    llm_config_path=str(config_path),
                    output_dir=tmpdir,
                )

            generated_path = Path(result["generated_path"])
            raw_responses_path = Path(result["raw_responses_path"])

            self.assertTrue(generated_path.exists())
            self.assertTrue(raw_responses_path.exists())

            generated_lines = [
                json.loads(line)
                for line in generated_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([item["prompt"] for item in generated_lines], ["alpha", "beta"])

            raw_lines = [
                json.loads(line)
                for line in raw_responses_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(raw_lines), 1)
            self.assertEqual(raw_lines[0]["request_id"], 1)
            self.assertEqual(raw_lines[0]["raw_content"], response_text)

    def test_execute_request_plan_defaults_reasoning_effort_for_gpt5_models(self):
        module = load_module()

        request_plan = {
            "schema_version": "1.0",
            "dimension_count": 1,
            "selected_dimensions": ["text_error"],
            "total_target_prompts": 1,
            "batch_size_prompts_per_request": 1,
            "total_requests": 1,
            "jobs": [
                {
                    "request_id": 1,
                    "dimension": "text_error",
                    "priority": 1,
                    "tier": "Tier A",
                    "coverage_mode": "concentrated",
                    "macro_bucket": "commercial_text_media",
                    "requested_prompts": 1,
                    "chunk_index": 1,
                    "template_family": "text_literal_template",
                    "system_prompt": "system",
                    "user_prompt": "user",
                }
            ],
        }

        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='["only one"]'))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=fake_create,
                )
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "llm_config.json"
            config_path.write_text(
                json.dumps({"llm": {"api_key": "test-key", "model": "gpt-5"}}),
                encoding="utf-8",
            )

            with patch.object(module, "_create_openai_client", return_value=fake_client):
                module.execute_request_plan(
                    request_plan=request_plan,
                    llm_config_path=str(config_path),
                    output_dir=tmpdir,
                )

        self.assertEqual(captured["reasoning_effort"], "minimal")

    def test_build_user_prompt_prevents_preexisting_target_degradation_for_high_risk_templates(self):
        module = load_module()

        hand_prompt = module._build_user_prompt(
            {
                "dimension": "hand_malformation",
                "macro_bucket": "human_activity_interaction",
                "requested_prompts": 3,
                "template_family": "hands_visible_template",
            }
        )
        face_prompt = module._build_user_prompt(
            {
                "dimension": "face_asymmetry",
                "macro_bucket": "human_portrait",
                "requested_prompts": 3,
                "template_family": "portrait_face_template",
            }
        )

        self.assertIn("must not already exhibit the target degradation", hand_prompt)
        self.assertIn("Hands must appear natural, complete, and anatomically normal", hand_prompt)
        self.assertIn("must not already exhibit the target degradation", face_prompt)
        self.assertIn("face should remain natural, symmetric, and cleanly rendered", face_prompt)


if __name__ == "__main__":
    unittest.main()
