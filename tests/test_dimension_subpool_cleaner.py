from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLEANER_PATH = ROOT / "data_generation" / "scripts" / "dimension_subpool_cleaner.py"


def load_module():
    spec = spec_from_file_location("dimension_subpool_cleaner", CLEANER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DimensionSubpoolCleanerTests(unittest.TestCase):
    def test_cleaner_script_exists(self):
        self.assertTrue(CLEANER_PATH.exists())

    def test_cleaner_filters_all_dimensions_and_preserves_full_index(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            (input_dir / "blur.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "clean full-body woman walking in studio light"}),
                        json.dumps({"prompt": "cute dragon Disney"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "context_mismatch.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "serene beach picnic with friends and soft sunset light"}),
                        json.dumps({"prompt": "the end the end the end the end the end the end the end"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            index_payload = {
                "dimensions": {
                    "blur": {
                        "count": 2,
                        "filename": str((input_dir / "blur.jsonl").resolve()),
                    },
                    "context_mismatch": {
                        "count": 2,
                        "filename": str((input_dir / "context_mismatch.jsonl").resolve()),
                    },
                }
            }
            index_path = input_dir / "index.json"
            index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            outputs = module.clean_dimension_subpools(
                input_index_path=str(index_path),
                output_dir=str(tmp / "out"),
            )

            cleaned_index = json.loads(Path(outputs["index_path"]).read_text(encoding="utf-8"))
            self.assertEqual(set(cleaned_index["dimensions"].keys()), {"blur", "context_mismatch"})
            self.assertEqual(cleaned_index["dimensions"]["blur"]["count"], 1)
            self.assertEqual(cleaned_index["dimensions"]["context_mismatch"]["count"], 1)

            blur_lines = [
                json.loads(line)
                for line in (Path(outputs["output_dir"]) / "blur.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            context_lines = [
                json.loads(line)
                for line in (Path(outputs["output_dir"]) / "context_mismatch.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["prompt"] for row in blur_lines], ["clean full-body woman walking in studio light"])
            self.assertEqual(
                [row["prompt"] for row in context_lines],
                ["serene beach picnic with friends and soft sunset light"],
            )


if __name__ == "__main__":
    unittest.main()
