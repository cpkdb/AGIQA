from importlib.util import module_from_spec, spec_from_file_location
import json
import random
import sys
import tempfile
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "data_generation" / "scripts" / "pipeline.py"


def load_pipeline_module():
    tools_pkg = types.ModuleType("tools")
    tools_pkg.prompt_degrader = types.ModuleType("prompt_degrader")
    tools_pkg.image_generator = types.ModuleType("image_generator")
    tools_pkg.degradation_judge = types.ModuleType("degradation_judge")
    sys.modules.setdefault("tools", tools_pkg)
    sys.modules.setdefault("tools.prompt_degrader", tools_pkg.prompt_degrader)
    sys.modules.setdefault("tools.image_generator", tools_pkg.image_generator)
    sys.modules.setdefault("tools.degradation_judge", tools_pkg.degradation_judge)

    spec = spec_from_file_location("pipeline_under_test", PIPELINE_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PipelineDimensionCompatibilityTests(unittest.TestCase):
    def test_subpool_sampling_skips_runtime_compat_filter_by_default(self):
        module = load_pipeline_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            subpool = tmp / "face_asymmetry.jsonl"
            subpool.write_text(
                json.dumps(
                    {
                        "prompt": "full-body woman walking in clean studio light",
                        "semantic_tags": ["has_person"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            pipeline = module.DataGenerationPipeline(output_dir=str(tmp / "out"))
            pipeline.configure_dimension_subpools(
                subpool_dir=tmp,
                dimension_registry={
                    "face_asymmetry": {
                        "filename": "face_asymmetry.jsonl",
                        "count": 1,
                        "skip_runtime_compat_filter": True,
                    }
                },
            )

            sampled = pipeline._sample_from_dimension_subpool(
                "face_asymmetry",
                sample_size=5,
                rng=random.Random(0),
            )

            prompts = [item["prompt"] for item in sampled]
            self.assertEqual(
                prompts,
                ["full-body woman walking in clean studio light"],
            )

    def test_object_structure_error_subpool_is_trusted_without_runtime_filter(self):
        module = load_pipeline_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            subpool = tmp / "object_structure_error.jsonl"
            subpool.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "elon musk pregnant with twins",
                                "semantic_tags": ["has_multiple_objects"],
                            }
                        ),
                        json.dumps(
                            {
                                "prompt": "landscape photography. house in the mountains, wes anderson film screenshot",
                                "semantic_tags": ["has_indoor_scene", "has_multiple_objects", "has_structured_object"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            pipeline = module.DataGenerationPipeline(output_dir=str(tmp / "out"))
            pipeline.configure_dimension_subpools(
                subpool_dir=tmp,
                dimension_registry={
                    "object_structure_error": {
                        "filename": "object_structure_error.jsonl",
                        "count": 2,
                    }
                },
            )

            sampled = pipeline._sample_from_dimension_subpool(
                "object_structure_error",
                sample_size=5,
                rng=random.Random(0),
            )

            prompts = [item["prompt"] for item in sampled]
            self.assertEqual(
                prompts,
                [
                    "elon musk pregnant with twins",
                    "landscape photography. house in the mountains, wes anderson film screenshot",
                ],
            )

    def test_animal_anatomy_error_subpool_is_trusted_without_runtime_filter(self):
        module = load_pipeline_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            subpool = tmp / "animal_anatomy_error.jsonl"
            subpool.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt": "emma watson wearing a futuristic metal kimono, half body portrait, sharp details, pale skin, warm lighting",
                                "semantic_tags": ["has_animal", "has_face", "has_person"],
                            }
                        ),
                        json.dumps(
                            {
                                "prompt": "an anatomical drawing of a wolf in profile with detailed fur and bone structure",
                                "semantic_tags": ["has_animal"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            pipeline = module.DataGenerationPipeline(output_dir=str(tmp / "out"))
            pipeline.configure_dimension_subpools(
                subpool_dir=tmp,
                dimension_registry={
                    "animal_anatomy_error": {
                        "filename": "animal_anatomy_error.jsonl",
                        "count": 2,
                    }
                },
            )

            sampled = pipeline._sample_from_dimension_subpool(
                "animal_anatomy_error",
                sample_size=5,
                rng=random.Random(0),
            )

            prompts = [item["prompt"] for item in sampled]
            self.assertEqual(
                prompts,
                [
                    "emma watson wearing a futuristic metal kimono, half body portrait, sharp details, pale skin, warm lighting",
                    "an anatomical drawing of a wolf in profile with detailed fur and bone structure",
                ],
            )


if __name__ == "__main__":
    unittest.main()
