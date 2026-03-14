from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "data_generation" / "scripts" / "shared_prompt_pool_family_builder.py"


def load_module():
    spec = spec_from_file_location("shared_prompt_pool_family_builder", BUILDER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SharedPromptPoolFamilyBuilderTests(unittest.TestCase):
    def test_builder_script_exists(self):
        self.assertTrue(BUILDER_PATH.exists())

    def test_human_full_body_realistic_builder_writes_standardized_candidates_and_screening_input(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_prompts = root / "merged_working_pool_cleaned_v1.jsonl"
            output_root = root / "output"

            records = [
                {
                    "prompt": "full-body photo of a smiling woman standing on a studio floor",
                    "semantic_tags": ["has_person"],
                    "signature": {"has_person": True, "has_full_body": True},
                },
                {
                    "prompt": "a realistic man standing in a hallway, full length portrait",
                    "semantic_tags": ["has_person"],
                    "signature": {"has_person": True, "has_full_body": False},
                },
                {
                    "prompt": "portrait close-up of a woman smiling softly",
                    "semantic_tags": ["has_person", "has_face"],
                    "signature": {"has_person": True, "has_full_body": False},
                },
                {
                    "prompt": "an armored knight in heavy armor standing in battle",
                    "semantic_tags": ["has_person"],
                    "signature": {"has_person": True, "has_full_body": True},
                },
                {
                    "prompt": "two women standing together in a crowd",
                    "semantic_tags": ["has_person"],
                    "signature": {"has_person": True, "has_full_body": True},
                },
                {
                    "prompt": "a sensual woman lying in bed, full body view",
                    "semantic_tags": ["has_person"],
                    "signature": {"has_person": True, "has_full_body": True},
                },
                {
                    "prompt": "a superhero woman flying across the sky",
                    "semantic_tags": ["has_person"],
                    "signature": {"has_person": True, "has_full_body": True},
                },
                {
                    "prompt": "a dog running in the park",
                    "semantic_tags": ["has_animal"],
                    "signature": {"has_person": False, "has_full_body": False},
                },
            ]
            with source_prompts.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            screening_spec_path = root / "prompt_pool_screening_spec_v1.json"
            screening_spec_path.write_text(
                json.dumps(
                    {
                        "shared_pool_families": {
                            "human_full_body_realistic": {
                                "builder_mode": "rule_recall_then_llm_screen",
                                "base_tags": ["has_person"],
                                "screening_goal": "保留可作为真实全身人物父池候选的 prompt。",
                                "llm_prompt_focus": [
                                    "single human subject",
                                    "readable full body or body proportion cues",
                                ],
                            }
                        },
                        "dimension_overrides": {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            build_targets_path = root / "prompt_pool_build_targets_v1.json"
            build_targets_path.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "shared_pool_family_outputs": {
                            "common": {
                                "directory_name": "shared_family_screened_pools_cleaned_v1",
                                "index_name": "index.json",
                                "candidate_suffix": "_candidates.jsonl",
                                "screening_input_suffix": "_screening_input.jsonl",
                            },
                            "turbo": {
                                "directory_name": "sd35_turbo_shared_family_screened_pools_clipsafe_v1",
                                "index_name": "index.json",
                                "candidate_suffix": "_candidates.jsonl",
                                "screening_input_suffix": "_screening_input.jsonl",
                            },
                        },
                        "dimension_subpool_outputs": {
                            "common": {"directory_name": "semantic_screened_dimension_subpools_cleaned_v1", "index_name": "index.json"},
                            "turbo": {"directory_name": "sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1", "index_name": "index.json"},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            outputs = module.write_shared_prompt_pool_family(
                source_prompts_path=str(source_prompts),
                output_root=str(output_root),
                family_name="human_full_body_realistic",
                pool_variant="common",
                screening_spec_path=str(screening_spec_path),
                build_targets_path=str(build_targets_path),
            )

            candidate_path = Path(outputs["candidate_path"])
            screening_input_path = Path(outputs["screening_input_path"])
            index_path = Path(outputs["index_path"])

            self.assertTrue(candidate_path.exists())
            self.assertTrue(screening_input_path.exists())
            self.assertTrue(index_path.exists())
            self.assertEqual(candidate_path.parent.name, "shared_family_screened_pools_cleaned_v1")
            self.assertEqual(candidate_path.name, "human_full_body_realistic_candidates.jsonl")
            self.assertEqual(screening_input_path.name, "human_full_body_realistic_screening_input.jsonl")

            candidate_rows = [
                json.loads(line)
                for line in candidate_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            kept_prompts = [row["prompt"] for row in candidate_rows]
            self.assertEqual(
                kept_prompts,
                [
                    "full-body photo of a smiling woman standing on a studio floor",
                    "a realistic man standing in a hallway, full length portrait",
                ],
            )
            self.assertTrue(all(row["shared_pool_family"] == "human_full_body_realistic" for row in candidate_rows))

            screening_rows = [
                json.loads(line)
                for line in screening_input_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(screening_rows), 2)
            self.assertEqual(
                [row["prompt"] for row in screening_rows],
                kept_prompts,
            )
            self.assertTrue(all(row["family_name"] == "human_full_body_realistic" for row in screening_rows))
            self.assertTrue(all("sample_id" in row for row in screening_rows))
            self.assertTrue(all("heuristics" in row for row in screening_rows))

            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            family_meta = index_payload["families"]["human_full_body_realistic"]
            self.assertEqual(family_meta["count"], 2)
            self.assertEqual(family_meta["builder_mode"], "rule_recall_then_llm_screen")
            self.assertEqual(family_meta["base_tags"], ["has_person"])
            self.assertEqual(family_meta["excluded_reason_counts"]["insufficient_body_readability"], 1)
            self.assertEqual(
                family_meta["excluded_reason_counts"]["contains_non_natural_or_identity_obscuring_human_presentation"],
                2,
            )
            self.assertEqual(family_meta["excluded_reason_counts"]["contains_multi_person_signal"], 1)
            self.assertEqual(family_meta["excluded_reason_counts"]["contains_bed_or_adult_pose_signal"], 1)
            self.assertEqual(family_meta["excluded_reason_counts"]["missing_base_tags"], 1)

    def test_structured_object_primary_filters_biological_dominant_records(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_prompts = root / "merged_working_pool_cleaned_v1.jsonl"
            output_root = root / "output"

            records = [
                {
                    "prompt": "a ceramic mug on a wooden desk beside a notebook",
                    "semantic_tags": ["has_structured_object"],
                    "signature": {"has_structured_object": True},
                },
                {
                    "prompt": "a closed silver laptop on an office desk",
                    "semantic_tags": ["has_structured_object"],
                    "signature": {"has_structured_object": True},
                },
                {
                    "prompt": "a woman holding a mug in a cafe",
                    "semantic_tags": ["has_person", "has_structured_object"],
                    "signature": {"has_person": True, "has_structured_object": True},
                },
                {
                    "prompt": "a cat beside a food bowl on the floor",
                    "semantic_tags": ["has_animal", "has_structured_object"],
                    "signature": {"has_animal": True, "has_structured_object": True},
                },
                {
                    "prompt": "portrait of a smiling woman",
                    "semantic_tags": ["has_person"],
                    "signature": {"has_person": True},
                },
            ]
            with source_prompts.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            screening_spec_path = root / "prompt_pool_screening_spec_v1.json"
            screening_spec_path.write_text(
                json.dumps(
                    {
                        "shared_pool_families": {
                            "structured_object_primary": {
                                "builder_mode": "rule_recall_then_llm_screen",
                                "base_tags": ["has_structured_object"],
                                "screening_goal": "保留单个显著结构化非生物物体。",
                                "llm_prompt_focus": [
                                    "one dominant structured non-living object",
                                    "shape and material cues are readable",
                                ],
                            }
                        },
                        "dimension_overrides": {
                            "object_structure_error": {"shared_pool_family": "structured_object_primary"},
                            "material_mismatch": {"shared_pool_family": "structured_object_primary"},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            build_targets_path = root / "prompt_pool_build_targets_v1.json"
            build_targets_path.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "shared_pool_family_outputs": {
                            "common": {
                                "directory_name": "shared_family_screened_pools_cleaned_v1",
                                "index_name": "index.json",
                                "candidate_suffix": "_candidates.jsonl",
                                "screening_input_suffix": "_screening_input.jsonl",
                            }
                        },
                        "dimension_subpool_outputs": {
                            "common": {"directory_name": "semantic_screened_dimension_subpools_cleaned_v1", "index_name": "index.json"}
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            outputs = module.write_shared_prompt_pool_family(
                source_prompts_path=str(source_prompts),
                output_root=str(output_root),
                family_name="structured_object_primary",
                pool_variant="common",
                screening_spec_path=str(screening_spec_path),
                build_targets_path=str(build_targets_path),
            )

            candidate_rows = [
                json.loads(line)
                for line in Path(outputs["candidate_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [row["prompt"] for row in candidate_rows],
                [
                    "a ceramic mug on a wooden desk beside a notebook",
                    "a closed silver laptop on an office desk",
                ],
            )

            index_payload = json.loads(Path(outputs["index_path"]).read_text(encoding="utf-8"))
            family_meta = index_payload["families"]["structured_object_primary"]
            self.assertEqual(family_meta["count"], 2)
            self.assertEqual(
                family_meta["excluded_reason_counts"]["contains_biological_primary_signal"],
                2,
            )
            self.assertEqual(
                family_meta["excluded_reason_counts"]["missing_base_tags"],
                1,
            )
            self.assertEqual(
                family_meta["target_dimensions"],
                ["material_mismatch", "object_structure_error"],
            )

    def test_multi_object_reference_keeps_broad_multi_object_candidates(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_prompts = root / "merged_working_pool_cleaned_v1.jsonl"
            output_root = root / "output"

            records = [
                {
                    "prompt": "a mug beside a laptop on a desk",
                    "semantic_tags": ["has_multiple_objects", "has_structured_object"],
                    "signature": {"has_multiple_objects": True, "has_structured_object": True},
                },
                {
                    "prompt": "two people standing next to a bicycle",
                    "semantic_tags": ["has_multiple_objects", "has_person"],
                    "signature": {"has_multiple_objects": True, "has_person": True},
                },
                {
                    "prompt": "a dog beside a metal bowl",
                    "semantic_tags": ["has_animal"],
                    "signature": {"has_animal": True},
                },
                {
                    "prompt": "a single chair in a studio",
                    "semantic_tags": ["has_structured_object"],
                    "signature": {"has_structured_object": True},
                },
                {
                    "prompt": "portrait close-up of a smiling woman",
                    "semantic_tags": ["has_person", "has_face"],
                    "signature": {"has_person": True, "has_face": True},
                },
            ]
            with source_prompts.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            screening_spec_path = root / "prompt_pool_screening_spec_v1.json"
            screening_spec_path.write_text(
                json.dumps(
                    {
                        "shared_pool_families": {
                            "multi_object_reference": {
                                "builder_mode": "rule_recall_then_llm_screen",
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
                        },
                        "dimension_overrides": {
                            "scale_inconsistency": {"shared_pool_family": "multi_object_reference"},
                            "penetration_overlap": {"shared_pool_family": "multi_object_reference"},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            build_targets_path = root / "prompt_pool_build_targets_v1.json"
            build_targets_path.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "shared_pool_family_outputs": {
                            "common": {
                                "directory_name": "shared_family_screened_pools_cleaned_v1",
                                "index_name": "index.json",
                                "candidate_suffix": "_candidates.jsonl",
                                "screening_input_suffix": "_screening_input.jsonl",
                            }
                        },
                        "dimension_subpool_outputs": {
                            "common": {"directory_name": "semantic_screened_dimension_subpools_cleaned_v1", "index_name": "index.json"}
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            outputs = module.write_shared_prompt_pool_family(
                source_prompts_path=str(source_prompts),
                output_root=str(output_root),
                family_name="multi_object_reference",
                pool_variant="common",
                screening_spec_path=str(screening_spec_path),
                build_targets_path=str(build_targets_path),
            )

            candidate_rows = [
                json.loads(line)
                for line in Path(outputs["candidate_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [row["prompt"] for row in candidate_rows],
                [
                    "a mug beside a laptop on a desk",
                    "two people standing next to a bicycle",
                    "a dog beside a metal bowl",
                ],
            )
            screening_rows = [
                json.loads(line)
                for line in Path(outputs["screening_input_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(all(row["target_dimensions"] == ["penetration_overlap", "scale_inconsistency"] for row in screening_rows))
            self.assertTrue(all("screening_goal" in row for row in screening_rows))
            self.assertTrue(all("llm_prompt_focus" in row for row in screening_rows))
            self.assertEqual(
                json.loads(Path(outputs["index_path"]).read_text(encoding="utf-8"))["families"]["multi_object_reference"]["excluded_reason_counts"]["insufficient_relation_or_reference_signal"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
