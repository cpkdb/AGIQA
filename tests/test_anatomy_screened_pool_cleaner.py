from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLEANER_PATH = ROOT / "data_generation" / "scripts" / "anatomy_screened_pool_cleaner.py"


def load_module():
    spec = spec_from_file_location("anatomy_screened_pool_cleaner", CLEANER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnatomyScreenedPoolCleanerTests(unittest.TestCase):
    def test_cleaner_script_exists(self):
        self.assertTrue(CLEANER_PATH.exists())

    def test_cleaner_filters_short_long_and_noisy_prompts(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            (input_dir / "face_asymmetry.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "clean full-body woman walking in studio light"}),
                        json.dumps({"prompt": "cute dragon Disney"}),
                        json.dumps({"prompt": "the end the end the end the end the end the end the end"}),
                        json.dumps({"prompt": "word " * 130}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            index_payload = {
                "dimensions": {
                    "face_asymmetry": {
                        "count": 4,
                        "filename": str((input_dir / "face_asymmetry.jsonl").resolve()),
                        "skip_runtime_compat_filter": True,
                    },
                    "shadow_mismatch": {
                        "count": 1,
                        "filename": "/abs/base/shadow_mismatch.jsonl",
                    },
                }
            }
            index_path = input_dir / "index.json"
            index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            outputs = module.clean_anatomy_screened_subpools(
                input_index_path=str(index_path),
                output_dir=str(tmp / "out"),
            )

            cleaned_index = json.loads(Path(outputs["index_path"]).read_text(encoding="utf-8"))
            self.assertEqual(cleaned_index["dimensions"]["face_asymmetry"]["count"], 1)
            self.assertEqual(cleaned_index["dimensions"]["shadow_mismatch"]["filename"], "/abs/base/shadow_mismatch.jsonl")
            self.assertTrue(cleaned_index["dimensions"]["face_asymmetry"]["skip_runtime_compat_filter"])

            cleaned_path = Path(outputs["output_dir"]) / "face_asymmetry.jsonl"
            lines = [json.loads(line) for line in cleaned_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["prompt"] for row in lines], ["clean full-body woman walking in studio light"])


if __name__ == "__main__":
    unittest.main()
