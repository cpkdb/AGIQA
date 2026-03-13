from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "data_generation" / "scripts" / "targeted_dimension_pool_builder.py"


def load_module():
    spec = spec_from_file_location("targeted_dimension_pool_builder", BUILDER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TargetedDimensionPoolBuilderTests(unittest.TestCase):
    def test_builder_script_exists(self):
        self.assertTrue(BUILDER_PATH.exists())

    def test_build_targeted_dimension_subpools_reuses_existing_cleaned_pool_sources(self):
        module = load_module()

        source_records = [
            {"prompt": "a wooden chair in a bright room", "semantic_tags": ["has_structured_object"]},
            {"prompt": "a glass bottle on a wooden table", "semantic_tags": ["has_structured_object"]},
            {"prompt": "clear water in a glass aquarium", "semantic_tags": ["has_structured_object"]},
            {"prompt": "a portrait of a woman smiling softly", "semantic_tags": ["has_face"]},
        ]
        object_shape_records = [
            {"prompt": "a wooden chair in a bright room", "semantic_tags": ["has_structured_object"]},
            {"prompt": "a glass bottle on a wooden table", "semantic_tags": ["has_structured_object"]},
        ]

        result = module.build_targeted_dimension_subpools(
            source_records=source_records,
            object_shape_records=object_shape_records,
        )

        self.assertEqual(result["index"]["object_structure_error"]["count"], 2)
        self.assertEqual(result["index"]["material_mismatch"]["count"], 2)
        self.assertTrue(result["index"]["object_structure_error"]["skip_runtime_compat_filter"])
        self.assertTrue(result["index"]["material_mismatch"]["skip_runtime_compat_filter"])

        self.assertEqual(
            [item["prompt"] for item in result["subpools"]["material_mismatch"]],
            ["a wooden chair in a bright room", "a glass bottle on a wooden table"],
        )
        self.assertEqual(
            [item["prompt"] for item in result["subpools"]["object_structure_error"]],
            ["a wooden chair in a bright room", "a glass bottle on a wooden table"],
        )


if __name__ == "__main__":
    unittest.main()
