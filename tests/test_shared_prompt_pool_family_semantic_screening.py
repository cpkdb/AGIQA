from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import unittest
import json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "data_generation" / "scripts" / "shared_prompt_pool_family_semantic_screening.py"


def load_module():
    spec = spec_from_file_location("shared_prompt_pool_family_semantic_screening", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SharedPromptPoolFamilySemanticScreeningTests(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists())

    def test_prepare_family_candidates_uses_base_tag_recall_only(self):
        module = load_module()

        records = [
            {
                "prompt": "a single woman standing in a hallway",
                "semantic_tags": ["has_person"],
                "signature": {"has_person": True, "has_full_body": True},
            },
            {
                "prompt": "two people talking in a cafe",
                "semantic_tags": ["has_person", "has_multiple_objects"],
                "signature": {"has_person": True},
            },
            {
                "prompt": "portrait close-up of a smiling woman",
                "semantic_tags": ["has_person", "has_face"],
                "signature": {"has_person": True},
            },
            {
                "prompt": "a ceramic mug on a wooden desk",
                "semantic_tags": ["has_structured_object"],
                "signature": {"has_structured_object": True},
            },
            {
                "prompt": "two people talking in a cafe",
                "semantic_tags": ["has_person"],
                "signature": {"has_person": True},
            },
        ]
        family_spec = {
            "base_tags": ["has_person"],
            "screening_goal": "保留至少一个可读主人物的人体场景。",
            "llm_prompt_focus": [
                "at least one readable main human subject",
                "body proportion cues should be readable",
            ],
        }
        target_dimensions = ["body_proportion_error", "extra_limbs"]

        candidates = module.prepare_family_candidates(
            source_records=records,
            family_name="human_full_body_realistic",
            family_spec=family_spec,
            target_dimensions=target_dimensions,
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            [item["prompt"] for item in candidates],
            [
                "a single woman standing in a hallway",
                "two people talking in a cafe",
                "portrait close-up of a smiling woman",
            ],
        )
        self.assertTrue(all(item["family_name"] == "human_full_body_realistic" for item in candidates))
        self.assertTrue(all(item["base_tags"] == ["has_person"] for item in candidates))
        self.assertTrue(all(item["target_dimensions"] == target_dimensions for item in candidates))
        self.assertTrue(all("sample_id" in item for item in candidates))

    def test_select_candidate_shard_partitions_stably(self):
        module = load_module()

        candidates = [
            {"sample_id": "s1", "prompt": "p1"},
            {"sample_id": "s2", "prompt": "p2"},
            {"sample_id": "s3", "prompt": "p3"},
            {"sample_id": "s4", "prompt": "p4"},
            {"sample_id": "s5", "prompt": "p5"},
        ]

        shard0 = module.select_candidate_shard(candidates, num_shards=2, shard_index=0)
        shard1 = module.select_candidate_shard(candidates, num_shards=2, shard_index=1)

        self.assertEqual([row["sample_id"] for row in shard0], ["s1", "s3", "s5"])
        self.assertEqual([row["sample_id"] for row in shard1], ["s2", "s4"])
        self.assertEqual(
            sorted([row["sample_id"] for row in shard0 + shard1]),
            ["s1", "s2", "s3", "s4", "s5"],
        )

    def test_select_candidate_shard_validates_args(self):
        module = load_module()

        with self.assertRaises(ValueError):
            module.select_candidate_shard([], num_shards=0, shard_index=0)
        with self.assertRaises(ValueError):
            module.select_candidate_shard([], num_shards=2, shard_index=2)

    def test_build_prompts_are_binary_and_allow_multi_person_with_main_subject(self):
        module = load_module()

        family_spec = {
            "screening_goal": "保留至少一个可读主人物的人体场景。",
            "llm_prompt_focus": [
                "at least one readable main human subject",
                "body or limb cues should be readable",
                "multi-person scenes are allowed if one main subject is still clearly readable",
            ],
        }
        system_prompt = module.build_family_system_prompt("human_full_body_realistic")
        self.assertIn("Use binary labels only", system_prompt)
        self.assertIn("pass", system_prompt)
        self.assertIn("fail", system_prompt)
        self.assertIn("multiple people", system_prompt)
        self.assertIn("at least one readable main human subject", system_prompt)

        user_prompt = module.build_family_user_prompt(
            family_name="human_full_body_realistic",
            family_spec=family_spec,
            target_dimensions=["body_proportion_error", "extra_limbs"],
            batch=[
                {
                    "sample_id": "human_full_body_realistic-00001-abc",
                    "prompt": "two people talking in a cafe",
                },
                {
                    "sample_id": "human_full_body_realistic-00002-def",
                    "prompt": "portrait close-up of a smiling woman",
                },
            ],
        )
        self.assertIn("Shared family: human_full_body_realistic", user_prompt)
        self.assertIn("Target dimensions: body_proportion_error, extra_limbs", user_prompt)
        self.assertIn("Pass criteria:", user_prompt)
        self.assertIn("Fail criteria:", user_prompt)
        self.assertIn("multi-person scenes are allowed", user_prompt)
        self.assertIn('"label":"pass|fail"', user_prompt)

    def test_build_prompts_for_structured_object_primary_are_family_aware(self):
        module = load_module()

        family_spec = {
            "screening_goal": "保留有单个主导性结构化非生物物体的 prompt。",
            "llm_prompt_focus": [
                "one dominant structured non-living object",
                "shape, parts, and surface material are easy to perceive",
            ],
        }
        system_prompt = module.build_family_system_prompt("structured_object_primary")
        self.assertIn("structured_object_primary", system_prompt)
        self.assertIn("non-living object", system_prompt)

        user_prompt = module.build_family_user_prompt(
            family_name="structured_object_primary",
            family_spec=family_spec,
            target_dimensions=["object_structure_error", "material_mismatch"],
            batch=[
                {
                    "sample_id": "structured_object_primary-00001-abc",
                    "prompt": "a ceramic kettle on a wooden table",
                }
            ],
        )
        self.assertIn("Shared family: structured_object_primary", user_prompt)
        self.assertIn("Target dimensions: object_structure_error, material_mismatch", user_prompt)
        self.assertIn("dominant structured non-living object", user_prompt)
        self.assertIn("humans or animals as the main subject", user_prompt)
        self.assertIn('"family_name":"structured_object_primary"', user_prompt)

    def test_build_prompts_for_multi_object_reference_are_family_aware(self):
        module = load_module()

        family_spec = {
            "screening_goal": "保留至少两个可比较对象、或存在清楚对象关系和参照关系的 prompt。",
            "llm_prompt_focus": [
                "at least two comparable objects, or one object with a clear reference context",
                "object boundaries or object relations are readable",
            ],
        }
        system_prompt = module.build_family_system_prompt("multi_object_reference")
        self.assertIn("multi_object_reference", system_prompt)
        self.assertIn("two readable entities", system_prompt)

        user_prompt = module.build_family_user_prompt(
            family_name="multi_object_reference",
            family_spec=family_spec,
            target_dimensions=["scale_inconsistency", "penetration_overlap"],
            batch=[
                {
                    "sample_id": "multi_object_reference-00001-abc",
                    "prompt": "a mug beside a laptop on a desk",
                }
            ],
        )
        self.assertIn("Shared family: multi_object_reference", user_prompt)
        self.assertIn("Target dimensions: scale_inconsistency, penetration_overlap", user_prompt)
        self.assertIn("at least two readable entities", user_prompt)
        self.assertIn("single isolated subject", user_prompt)
        self.assertIn('"family_name":"multi_object_reference"', user_prompt)
        self.assertIn("objects, humans, or animals", user_prompt)

    def test_prepare_family_candidates_for_multi_object_reference_uses_light_prefilter(self):
        module = load_module()

        records = [
            {
                "prompt": "a mug beside a laptop on a desk",
                "semantic_tags": ["has_multiple_objects", "has_structured_object"],
                "signature": {"has_multiple_objects": True, "has_structured_object": True},
            },
            {
                "prompt": "a dog beside a metal bowl",
                "semantic_tags": ["has_animal"],
                "signature": {"has_animal": True},
            },
            {
                "prompt": "portrait close-up of a smiling woman",
                "semantic_tags": ["has_person", "has_face"],
                "signature": {"has_person": True, "has_face": True},
            },
            {
                "prompt": "a single chair in a studio",
                "semantic_tags": ["has_structured_object"],
                "signature": {"has_structured_object": True},
            },
        ]
        family_spec = {
            "base_tags": ["has_multiple_objects"],
            "prefilter_tags_any": [
                "has_multiple_objects",
                "has_person",
                "has_animal",
                "has_structured_object",
                "has_countable_objects",
            ],
            "screening_goal": "保留多个可读实体，或一个主体加清楚参照关系的 prompt。",
            "llm_prompt_focus": [
                "entities may be objects, humans, or animals",
                "readable relations or references are preferred",
            ],
        }

        candidates = module.prepare_family_candidates(
            source_records=records,
            family_name="multi_object_reference",
            family_spec=family_spec,
            target_dimensions=["scale_inconsistency", "penetration_overlap"],
        )

        self.assertEqual(
            [item["prompt"] for item in candidates],
            [
                "a mug beside a laptop on a desk",
                "a dog beside a metal bowl",
            ],
        )

    def test_write_family_screening_inputs_writes_manifest_and_input_jsonl(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_prompts = root / "source.jsonl"
            output_dir = root / "out"
            screening_spec_path = root / "spec.json"
            source_rows = [
                {"prompt": "a single woman standing in a hallway", "semantic_tags": ["has_person"], "signature": {"has_person": True}},
                {"prompt": "two people talking in a cafe", "semantic_tags": ["has_person"], "signature": {"has_person": True}},
                {"prompt": "a ceramic mug on a desk", "semantic_tags": ["has_structured_object"], "signature": {"has_structured_object": True}},
            ]
            source_prompts.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
                encoding="utf-8",
            )
            screening_spec_path.write_text(
                json.dumps(
                    {
                        "shared_pool_families": {
                            "human_full_body_realistic": {
                                "base_tags": ["has_person"],
                                "screening_goal": "保留至少一个可读主人物的人体场景。",
                                "llm_prompt_focus": ["at least one readable main human subject"],
                            }
                        },
                        "dimension_overrides": {
                            "body_proportion_error": {"shared_pool_family": "human_full_body_realistic"},
                            "extra_limbs": {"shared_pool_family": "human_full_body_realistic"},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            outputs = module.write_family_screening_inputs(
                source_prompts_path=str(source_prompts),
                output_dir=str(output_dir),
                family_name="human_full_body_realistic",
                screening_spec_path=str(screening_spec_path),
            )

            input_jsonl = Path(outputs["input_jsonl"])
            manifest_json = Path(outputs["manifest_json"])
            self.assertTrue(input_jsonl.exists())
            self.assertTrue(manifest_json.exists())
            rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["family_name"], "human_full_body_realistic")
            self.assertEqual(manifest["candidate_count"], 2)
            self.assertEqual(manifest["target_dimensions"], ["body_proportion_error", "extra_limbs"])

    def test_parse_family_results_recovers_valid_items_from_partially_malformed_json(self):
        module = load_module()

        raw_text = """```json
{"family_name":"structured_object_primary","results":[
{"sample_id":"structured_object_primary-00001-aaa","label":"pass","reason":"dominant object is clear"},
{"sample_id":"structured_object_primary-00002-bbb","label":"fail","reason":"human subject dominates"},
{"sample_id":"structured_object_primary-00003-ccc","label":"pass","reason":"broken "quoted" reason"}
]}
```"""

        items, parse_error = module.parse_family_results(raw_text)

        self.assertIsNotNone(parse_error)
        self.assertEqual(
            items,
            [
                {
                    "sample_id": "structured_object_primary-00001-aaa",
                    "label": "pass",
                    "reason": "dominant object is clear",
                },
                {
                    "sample_id": "structured_object_primary-00002-bbb",
                    "label": "fail",
                    "reason": "human subject dominates",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
