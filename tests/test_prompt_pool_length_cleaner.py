from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLEANER_PATH = ROOT / "data_generation" / "scripts" / "prompt_pool_length_cleaner.py"


def load_module():
    spec = spec_from_file_location("prompt_pool_length_cleaner", CLEANER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptPoolLengthCleanerTests(unittest.TestCase):
    def test_cleaner_script_exists(self):
        self.assertTrue(CLEANER_PATH.exists())

    def test_clean_records_filters_short_long_and_noisy(self):
        module = load_module()
        records = [
            {"prompt": "clean full body woman walking in studio light"},
            {"prompt": "cute dragon Disney"},
            {"prompt": "the end the end the end the end the end the end the end"},
            {"prompt": "word " * 130},
            {"prompt": "https://example.com image prompt ref"},
        ]

        cleaned, _dropped, summary = module.clean_records(records)

        self.assertEqual([row["prompt"] for row in cleaned], ["clean full body woman walking in studio light"])
        self.assertEqual(summary["kept"], 1)
        self.assertEqual(summary["short"], 2)
        self.assertEqual(summary["long"], 1)
        self.assertEqual(summary["noise"], 1)

    def test_clean_prompt_pool_writes_expected_outputs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "merged.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "clean full body woman walking in studio light"}),
                        json.dumps({"prompt": "cute dragon Disney"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            outputs = module.clean_prompt_pool(
                input_path=str(input_path),
                output_dir=str(tmp / "out"),
            )

            cleaned_path = Path(outputs["cleaned_path"])
            summary_path = Path(outputs["summary_path"])
            dropped_path = Path(outputs["dropped_path"])

            self.assertTrue(cleaned_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(dropped_path.exists())

            cleaned_lines = [json.loads(line) for line in cleaned_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["prompt"] for row in cleaned_lines], ["clean full body woman walking in studio light"])


if __name__ == "__main__":
    unittest.main()
