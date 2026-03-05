from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "data_generation" / "scripts" / "dimension_subpool_builder.py"


def load_module():
    spec = spec_from_file_location("dimension_subpool_builder", BUILDER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DimensionSubpoolBuilderTests(unittest.TestCase):
    def test_dimension_subpool_builder_script_exists(self):
        self.assertTrue(BUILDER_PATH.exists())

    def test_build_dimension_subpools_filters_by_compatibility(self):
        module = load_module()

        records = [
            {
                "record_id": 1,
                "prompt": 'a woman holding a sign saying "OPEN"',
                "semantic_tags": ["has_person", "has_hand", "has_text", "has_quoted_text"],
            },
            {
                "record_id": 2,
                "prompt": "a smiling portrait of a man",
                "semantic_tags": ["has_person", "has_face"],
            },
            {
                "record_id": 3,
                "prompt": "a lake reflection at sunset",
                "semantic_tags": ["has_reflective_surface"],
            },
        ]

        result = module.build_dimension_subpools(
            records,
            dimensions=["hand_malformation", "face_asymmetry", "text_error"],
        )

        self.assertEqual(result["index"]["hand_malformation"]["count"], 1)
        self.assertEqual(result["index"]["face_asymmetry"]["count"], 1)
        self.assertEqual(result["index"]["text_error"]["count"], 1)
        self.assertEqual(len(result["subpools"]["text_error"]), 1)
        self.assertEqual(result["subpools"]["text_error"][0]["prompt"], 'a woman holding a sign saying "OPEN"')

    def test_write_dimension_subpools_creates_index_and_files(self):
        module = load_module()

        records = [
            {
                "record_id": 1,
                "prompt": 'a woman holding a sign saying "OPEN"',
                "semantic_tags": ["has_person", "has_hand", "has_text", "has_quoted_text"],
            },
            {
                "record_id": 2,
                "prompt": "a smiling portrait of a man",
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

            outputs = module.write_dimension_subpools(
                input_path=str(input_path),
                output_dir=str(tmpdir_path),
                dimensions=["hand_malformation", "face_asymmetry"],
            )

            index_path = Path(outputs["index_path"])
            self.assertTrue(index_path.exists())

            subpool_dir = Path(outputs["subpool_dir"])
            hand_path = subpool_dir / "hand_malformation.jsonl"
            face_path = subpool_dir / "face_asymmetry.jsonl"
            self.assertTrue(hand_path.exists())
            self.assertTrue(face_path.exists())

            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(index["dimensions"]["hand_malformation"]["count"], 1)
            self.assertEqual(index["dimensions"]["face_asymmetry"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
