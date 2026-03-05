from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = ROOT / "data_generation" / "scripts" / "prompt_candidate_sampler.py"


def load_module():
    spec = spec_from_file_location("prompt_candidate_sampler", SAMPLER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptCandidateSamplerTests(unittest.TestCase):
    def test_prompt_candidate_sampler_script_exists(self):
        self.assertTrue(SAMPLER_PATH.exists())

    def test_assign_bucket_identifies_specialized_bucket(self):
        module = load_module()

        diffusion_record = {
            "source": "diffusiondb",
            "prompt": "a woman holding a sign saying OPEN",
            "semantic_tags": ["has_person", "has_hand", "has_text"],
            "signature": {"has_person": True, "has_hand": True, "has_text": True},
        }
        pick_record = {
            "source": "pick_a_pic_v2",
            "prompt": "two people hugging and smiling in a park",
            "semantic_tags": ["has_person", "has_multiple_objects"],
            "signature": {"has_person": True, "has_multiple_objects": True},
        }

        self.assertEqual(
            module.assign_bucket(diffusion_record, "diffusiondb"),
            "people_face_hands",
        )
        self.assertEqual(
            module.assign_bucket(pick_record, "pick_a_pic_v2"),
            "people_and_activities",
        )

    def test_sample_working_pool_applies_bucket_quotas_and_top_up(self):
        module = load_module()

        records = [
            {
                "source": "diffusiondb",
                "record_id": 1,
                "prompt": "a portrait of a woman holding a sign",
                "semantic_tags": ["has_person", "has_hand", "has_text"],
                "signature": {"has_person": True, "has_hand": True, "has_text": True},
            },
            {
                "source": "diffusiondb",
                "record_id": 2,
                "prompt": "three apples on a table",
                "semantic_tags": ["has_countable_objects", "has_multiple_objects"],
                "signature": {"has_countable_objects": True, "has_multiple_objects": True},
            },
            {
                "source": "diffusiondb",
                "record_id": 3,
                "prompt": "a scenic mountain lake reflection",
                "semantic_tags": ["has_reflective_surface"],
                "signature": {"has_reflective_surface": True},
            },
            {
                "source": "diffusiondb",
                "record_id": 4,
                "prompt": "a cat in a living room",
                "semantic_tags": ["has_animal", "has_indoor_scene"],
                "signature": {"has_animal": True, "has_indoor_scene": True},
            },
            {
                "source": "diffusiondb",
                "record_id": 5,
                "prompt": "a city street at sunset",
                "semantic_tags": [],
                "signature": {},
            },
            {
                "source": "pick_a_pic_v2",
                "record_id": 6,
                "prompt": "two people dancing happily",
                "semantic_tags": ["has_person", "has_multiple_objects"],
                "signature": {"has_person": True, "has_multiple_objects": True},
            },
            {
                "source": "pick_a_pic_v2",
                "record_id": 7,
                "prompt": "a surreal floating castle above a forest",
                "semantic_tags": ["has_background"],
                "signature": {"has_background": True},
            },
            {
                "source": "t2i_compbench",
                "record_id": 8,
                "prompt": "two apples and one orange",
                "semantic_tags": ["has_countable_objects", "has_multiple_objects"],
                "signature": {"has_countable_objects": True, "has_multiple_objects": True},
            },
        ]

        plan = {
            "target_final_pool_size": 7,
            "planned_top_up_size": 1,
            "top_up_source": "diffusiondb",
            "sources": [
                {
                    "name": "diffusiondb",
                    "final_selected_size": 4,
                    "buckets": [
                        {"name": "people_face_hands", "final_selected_size": 1},
                        {"name": "count_constraints", "final_selected_size": 1},
                        {"name": "reflection_optics", "final_selected_size": 1},
                        {"name": "general_scenes", "final_selected_size": 1},
                    ],
                },
                {
                    "name": "pick_a_pic_v2",
                    "final_selected_size": 2,
                    "buckets": [
                        {"name": "people_and_activities", "final_selected_size": 1},
                        {"name": "creative_but_structured", "final_selected_size": 1},
                    ],
                },
                {
                    "name": "t2i_compbench",
                    "final_selected_size": 0,
                },
            ],
        }

        sampled, summary = module.sample_working_pool(records, plan=plan, seed=7)

        self.assertEqual(len(sampled), 7)
        self.assertEqual(summary["output_count"], 7)
        self.assertEqual(summary["source_counts"]["diffusiondb"], 5)
        self.assertEqual(summary["source_counts"]["pick_a_pic_v2"], 2)
        self.assertEqual(summary["top_up"]["source"], "diffusiondb")
        self.assertEqual(summary["top_up"]["selected_count"], 1)
        self.assertEqual(
            summary["bucket_counts"]["diffusiondb"]["people_face_hands"], 1
        )
        self.assertEqual(
            summary["bucket_counts"]["pick_a_pic_v2"]["people_and_activities"], 1
        )

    def test_sample_candidate_file_writes_outputs(self):
        module = load_module()

        records = [
            {
                "source": "parti_prompts",
                "record_id": 1,
                "prompt": "a red apple on a table",
                "semantic_tags": [],
                "signature": {},
            },
            {
                "source": "longbench_t2i",
                "record_id": 2,
                "prompt": "a long detailed indoor scene with multiple objects",
                "semantic_tags": ["has_indoor_scene", "has_multiple_objects"],
                "signature": {"has_indoor_scene": True, "has_multiple_objects": True},
            },
        ]

        plan = {
            "target_final_pool_size": 2,
            "planned_top_up_size": 0,
            "top_up_source": "diffusiondb",
            "sources": [
                {"name": "parti_prompts", "final_selected_size": 1},
                {"name": "longbench_t2i", "final_selected_size": 1},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in records
                ),
                encoding="utf-8",
            )

            outputs = module.sample_candidate_file(
                input_path=str(input_path),
                output_dir=str(tmpdir_path),
                plan=plan,
                seed=11,
            )

            sampled_path = Path(outputs["sampled_path"])
            summary_path = Path(outputs["summary_path"])
            self.assertTrue(sampled_path.exists())
            self.assertTrue(summary_path.exists())

            sampled_lines = sampled_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(sampled_lines), 2)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["output_count"], 2)
            self.assertEqual(summary["source_counts"]["parti_prompts"], 1)


if __name__ == "__main__":
    unittest.main()
