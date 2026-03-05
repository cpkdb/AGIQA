from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TAGGER_PATH = ROOT / "data_generation" / "scripts" / "prompt_candidate_tagger.py"


def load_module():
    spec = spec_from_file_location("prompt_candidate_tagger", TAGGER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptCandidateTaggerTests(unittest.TestCase):
    def test_prompt_candidate_tagger_script_exists(self):
        self.assertTrue(TAGGER_PATH.exists())

    def test_tag_candidates_adds_semantic_tags_and_signature(self):
        module = load_module()

        records = [
            {"source": "diffusiondb", "prompt": 'a portrait of a woman holding a sign saying "OPEN"'},
            {"source": "longbench_t2i", "prompt": "a lake reflection with mountains"},
        ]

        tagged, summary = module.tag_candidates(records)

        self.assertEqual(len(tagged), 2)
        self.assertIn("has_person", tagged[0]["semantic_tags"])
        self.assertIn("has_hand", tagged[0]["semantic_tags"])
        self.assertIn("has_text", tagged[0]["semantic_tags"])
        self.assertTrue(tagged[0]["signature"]["has_quoted_text"])
        self.assertIn("has_reflective_surface", tagged[1]["semantic_tags"])
        self.assertEqual(summary["input_count"], 2)
        self.assertEqual(summary["output_count"], 2)
        self.assertEqual(summary["tag_counts"]["has_person"], 1)
        self.assertEqual(summary["tag_counts"]["has_reflective_surface"], 1)

    def test_tag_candidate_file_writes_outputs(self):
        module = load_module()

        records = [
            {"source": "diffusiondb", "record_id": 1, "prompt": "a portrait of a man"},
            {"source": "parti_prompts", "record_id": 2, "prompt": "a mirror on a wall"},
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

            outputs = module.tag_candidate_file(
                input_path=str(input_path),
                output_dir=str(tmpdir_path),
            )

            tagged_path = Path(outputs["tagged_path"])
            summary_path = Path(outputs["summary_path"])
            self.assertTrue(tagged_path.exists())
            self.assertTrue(summary_path.exists())

            tagged_lines = tagged_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(tagged_lines), 2)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["input_count"], 2)
            self.assertEqual(summary["output_count"], 2)
            self.assertIn("has_person", summary["tag_counts"])


if __name__ == "__main__":
    unittest.main()
