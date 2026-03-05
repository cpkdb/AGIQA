from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data_generation" / "scripts" / "prompt_macro_taxonomy_report.py"


def load_module():
    spec = spec_from_file_location("prompt_macro_taxonomy_report", REPORT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptMacroTaxonomyReportTests(unittest.TestCase):
    def test_prompt_macro_taxonomy_report_script_exists(self):
        self.assertTrue(REPORT_PATH.exists())

    def test_assign_macro_category_uses_macro_scene_rules(self):
        module = load_module()

        portrait = {
            "prompt": "a close-up portrait of a smiling woman in studio lighting",
            "semantic_tags": ["has_person", "has_face"],
        }
        text_media = {
            "prompt": 'a storefront sign saying "OPEN" above a cafe entrance',
            "semantic_tags": ["has_text", "has_quoted_text", "has_logo_or_symbol"],
        }
        food = {
            "prompt": "a plate of sushi on a wooden dining table",
            "semantic_tags": [],
        }

        self.assertEqual(module.assign_macro_category(portrait), "human_portrait")
        self.assertEqual(module.assign_macro_category(text_media), "commercial_text_media")
        self.assertEqual(module.assign_macro_category(food), "food_tabletop")

    def test_build_macro_reports_creates_distribution_and_matrix(self):
        module = load_module()

        records = [
            {
                "prompt": 'a storefront sign saying "OPEN" above a cafe entrance',
                "semantic_tags": ["has_text", "has_quoted_text", "has_logo_or_symbol"],
            },
            {
                "prompt": "a close-up portrait of a smiling woman in studio lighting",
                "semantic_tags": ["has_person", "has_face"],
            },
            {
                "prompt": "two people dancing on a city street at night",
                "semantic_tags": ["has_person", "has_multiple_objects"],
            },
            {
                "prompt": "a calm lake reflection with mountains",
                "semantic_tags": ["has_reflective_surface"],
            },
        ]

        result = module.build_macro_reports(
            records,
            dimensions=["text_error", "face_asymmetry", "reflection_error"],
        )

        self.assertEqual(result["distribution"]["total_prompts"], 4)
        self.assertEqual(
            result["distribution"]["categories"]["commercial_text_media"]["count"], 1
        )
        self.assertEqual(
            result["distribution"]["categories"]["human_portrait"]["count"], 1
        )
        self.assertEqual(
            result["matrix"]["dimensions"]["text_error"]["by_macro_category"]["commercial_text_media"],
            1,
        )
        self.assertEqual(
            result["matrix"]["dimensions"]["face_asymmetry"]["by_macro_category"]["human_portrait"],
            1,
        )
        self.assertEqual(
            result["matrix"]["dimensions"]["reflection_error"]["by_macro_category"]["natural_landscape"],
            1,
        )

    def test_write_macro_reports_writes_outputs(self):
        module = load_module()

        records = [
            {
                "prompt": 'a storefront sign saying "OPEN" above a cafe entrance',
                "semantic_tags": ["has_text", "has_quoted_text", "has_logo_or_symbol"],
            },
            {
                "prompt": "a close-up portrait of a smiling woman in studio lighting",
                "semantic_tags": ["has_person", "has_face"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.jsonl"
            input_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            outputs = module.write_macro_reports(
                input_path=str(input_path),
                output_dir=str(tmpdir_path),
                dimensions=["text_error", "face_asymmetry"],
            )

            tagged_path = Path(outputs["tagged_path"])
            dist_path = Path(outputs["distribution_path"])
            matrix_path = Path(outputs["matrix_path"])

            self.assertTrue(tagged_path.exists())
            self.assertTrue(dist_path.exists())
            self.assertTrue(matrix_path.exists())

            distribution = json.loads(dist_path.read_text(encoding="utf-8"))
            self.assertEqual(distribution["total_prompts"], 2)
            self.assertEqual(distribution["categories"]["commercial_text_media"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
