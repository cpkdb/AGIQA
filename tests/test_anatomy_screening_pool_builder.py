from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "data_generation" / "scripts" / "anatomy_screening_pool_builder.py"


def load_module():
    spec = spec_from_file_location("anatomy_screening_pool_builder", BUILDER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnatomyScreeningPoolBuilderTests(unittest.TestCase):
    def test_builder_script_exists(self):
        self.assertTrue(BUILDER_PATH.exists())

    def test_build_screened_dimension_subpools_uses_shared_core_and_deltas(self):
        module = load_module()

        source_records = [
            {
                "prompt": "body-pass",
                "semantic_tags": ["has_person"],
                "signature": {"has_person": True, "has_face": True, "has_full_body": True},
            },
            {
                "prompt": "body-uncertain",
                "semantic_tags": ["has_person"],
                "signature": {"has_person": True, "has_face": True, "has_full_body": True},
            },
            {
                "prompt": "face-delta",
                "semantic_tags": ["has_face"],
                "signature": {"has_person": True, "has_face": True, "has_full_body": False},
            },
            {
                "prompt": "hand-delta",
                "semantic_tags": ["has_hand"],
                "signature": {"has_person": True, "has_face": False, "has_full_body": False},
            },
            {"prompt": "animal-pass", "semantic_tags": ["has_animal"]},
            {"prompt": "animal-uncertain", "semantic_tags": ["has_animal"]},
        ]
        body_results = [
            {"dimension": "body_proportion_error", "prompt": "body-pass", "label": "pass"},
            {"dimension": "body_proportion_error", "prompt": "body-uncertain", "label": "uncertain"},
            {"dimension": "body_proportion_error", "prompt": "body-fail", "label": "fail"},
        ]
        hand_face_results = [
            {"dimension": "face_asymmetry", "prompt": "face-delta", "label": "pass"},
            {"dimension": "face_asymmetry", "prompt": "face-fail", "label": "fail"},
            {"dimension": "hand_malformation", "prompt": "hand-delta", "label": "uncertain"},
            {"dimension": "hand_malformation", "prompt": "hand-fail", "label": "fail"},
        ]
        animal_results = [
            {"dimension": "animal_anatomy_error", "prompt": "animal-pass", "label": "pass"},
            {"dimension": "animal_anatomy_error", "prompt": "animal-uncertain", "label": "uncertain"},
            {"dimension": "animal_anatomy_error", "prompt": "animal-fail", "label": "fail"},
        ]

        result = module.build_screened_dimension_subpools(
            source_records=source_records,
            body_results=body_results,
            hand_face_results=hand_face_results,
            animal_results=animal_results,
        )

        self.assertEqual(result["index"]["body_proportion_error"]["count"], 2)
        self.assertEqual(result["index"]["impossible_pose"]["count"], 2)
        self.assertEqual(result["index"]["extra_limbs"]["count"], 2)
        self.assertEqual(result["index"]["expression_mismatch"]["count"], 2)
        self.assertEqual(result["index"]["face_asymmetry"]["count"], 3)
        self.assertEqual(result["index"]["hand_malformation"]["count"], 3)
        self.assertEqual(result["index"]["animal_anatomy_error"]["count"], 2)

        self.assertEqual(
            [item["prompt"] for item in result["subpools"]["face_asymmetry"]],
            ["body-pass", "body-uncertain", "face-delta"],
        )
        self.assertEqual(
            [item["prompt"] for item in result["subpools"]["hand_malformation"]],
            ["body-pass", "body-uncertain", "hand-delta"],
        )
        self.assertEqual(
            [item["prompt"] for item in result["subpools"]["animal_anatomy_error"]],
            ["animal-pass", "animal-uncertain"],
        )

    def test_builder_filters_shared_human_core_by_dimension_specific_signature(self):
        module = load_module()

        source_records = [
            {
                "prompt": "human-face-only",
                "semantic_tags": ["has_person"],
                "signature": {"has_person": True, "has_face": True, "has_full_body": False},
            },
            {
                "prompt": "human-full-body",
                "semantic_tags": ["has_person"],
                "signature": {"has_person": True, "has_face": False, "has_full_body": True},
            },
            {
                "prompt": "bad-false-human",
                "semantic_tags": ["has_person"],
                "signature": {"has_person": True, "has_face": False, "has_full_body": False},
            },
        ]
        body_results = [
            {"dimension": "body_proportion_error", "prompt": "human-face-only", "label": "uncertain"},
            {"dimension": "body_proportion_error", "prompt": "human-full-body", "label": "uncertain"},
            {"dimension": "body_proportion_error", "prompt": "bad-false-human", "label": "uncertain"},
        ]

        result = module.build_screened_dimension_subpools(
            source_records=source_records,
            body_results=body_results,
            hand_face_results=[],
            animal_results=[],
        )

        self.assertEqual(
            [item["prompt"] for item in result["subpools"]["expression_mismatch"]],
            ["human-face-only"],
        )
        self.assertEqual(
            [item["prompt"] for item in result["subpools"]["body_proportion_error"]],
            ["human-full-body"],
        )
        self.assertEqual(
            [item["prompt"] for item in result["subpools"]["extra_limbs"]],
            ["human-full-body"],
        )


if __name__ == "__main__":
    unittest.main()
