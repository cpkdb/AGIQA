from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data_generation" / "scripts" / "prompt_pool_coverage_report.py"


def load_module():
    spec = spec_from_file_location("prompt_pool_coverage_report", REPORT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptPoolCoverageReportTests(unittest.TestCase):
    def test_prompt_pool_coverage_report_script_exists(self):
        self.assertTrue(REPORT_PATH.exists())

    def test_build_coverage_report_counts_compatible_prompts(self):
        module = load_module()

        records = [
            {
                "prompt": 'a woman holding a sign saying "OPEN"',
                "semantic_tags": ["has_person", "has_hand", "has_text", "has_quoted_text"],
            },
            {
                "prompt": "a smiling portrait of a man",
                "semantic_tags": ["has_person", "has_face"],
            },
            {
                "prompt": "a lake reflection at sunset",
                "semantic_tags": ["has_reflective_surface"],
            },
        ]
        tag_requirements = {
            "hand_malformation": {"required": ["has_hand"], "alternative": [], "preferred": []},
            "face_asymmetry": {"required": ["has_face"], "alternative": [], "preferred": []},
            "reflection_error": {"required": ["has_reflective_surface"], "alternative": [], "preferred": []},
            "text_error": {"required": ["has_text"], "alternative": [], "preferred": []},
        }

        report = module.build_coverage_report(records, tag_requirements=tag_requirements)

        self.assertEqual(report["total_prompts"], 3)
        self.assertEqual(report["dimensions"]["hand_malformation"]["available"], 1)
        self.assertEqual(report["dimensions"]["face_asymmetry"]["available"], 1)
        self.assertEqual(report["dimensions"]["reflection_error"]["available"], 1)
        self.assertEqual(report["dimensions"]["text_error"]["available"], 1)
        self.assertEqual(report["dimensions"]["text_error"]["gap_level"], "scarce")

    def test_write_coverage_report_writes_outputs(self):
        module = load_module()

        records = [
            {
                "prompt": 'a woman holding a sign saying "OPEN"',
                "semantic_tags": ["has_person", "has_hand", "has_text", "has_quoted_text"],
            },
            {
                "prompt": "a smiling portrait of a man",
                "semantic_tags": ["has_person", "has_face"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in records
                ),
                encoding="utf-8",
            )

            outputs = module.write_coverage_report(
                input_path=str(input_path),
                output_dir=str(tmpdir_path),
                dimensions=["hand_malformation", "face_asymmetry"],
            )

            report_path = Path(outputs["report_path"])
            self.assertTrue(report_path.exists())

            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(data["total_prompts"], 2)
            self.assertEqual(data["dimensions"]["hand_malformation"]["available"], 1)
            self.assertEqual(data["dimensions"]["face_asymmetry"]["available"], 1)


if __name__ == "__main__":
    unittest.main()
