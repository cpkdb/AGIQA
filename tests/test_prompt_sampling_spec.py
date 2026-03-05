from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "data_generation" / "data" / "prompt_sampling_spec.json"


class PromptSamplingSpecTests(unittest.TestCase):
    def test_sampling_spec_exists_for_primary_sources(self):
        self.assertTrue(SPEC_PATH.exists())

        data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            [source["name"] for source in data["sources"]],
            ["diffusiondb", "pick_a_pic_v2"],
        )

    def test_diffusiondb_bucket_plan_matches_target(self):
        data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        diffusiondb = data["sources"][0]

        self.assertEqual(diffusiondb["candidate_pool_size"], 60000)
        self.assertEqual(diffusiondb["final_selected_size"], 14000)
        self.assertEqual(
            sum(bucket["final_selected_size"] for bucket in diffusiondb["buckets"]),
            14000,
        )

    def test_pick_a_pic_bucket_plan_matches_target(self):
        data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        pick_a_pic = data["sources"][1]

        self.assertEqual(pick_a_pic["candidate_pool_size"], 30000)
        self.assertEqual(pick_a_pic["final_selected_size"], 9000)
        self.assertEqual(
            sum(bucket["final_selected_size"] for bucket in pick_a_pic["buckets"]),
            9000,
        )

    def test_buckets_define_prompt_side_signals_not_images(self):
        data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

        for source in data["sources"]:
            for bucket in source["buckets"]:
                self.assertIn("selection_signals", bucket)
                self.assertIsInstance(bucket["selection_signals"], list)
                self.assertNotIn("image_embedding", bucket["selection_signals"])


if __name__ == "__main__":
    unittest.main()
