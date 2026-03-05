from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data_generation" / "data" / "prompt_source_catalog.json"


class PromptSourceCatalogTests(unittest.TestCase):
    def test_catalog_exists_and_has_expected_sources(self):
        self.assertTrue(CATALOG_PATH.exists())

        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        sources = data["sources"]

        self.assertEqual(
            [source["name"] for source in sources],
            [
                "diffusiondb",
                "pick_a_pic_v2",
                "t2i_compbench",
                "parti_prompts",
                "longbench_t2i",
            ],
        )

    def test_catalog_totals_match_initial_public_only_plan(self):
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

        total_candidate = sum(source["candidate_pool_size"] for source in data["sources"])
        total_selected = sum(source["final_selected_size"] for source in data["sources"])

        self.assertEqual(total_candidate, 98132)
        self.assertEqual(total_selected, 29000)
        self.assertEqual(data["top_up_source"], "diffusiondb")
        self.assertEqual(data["target_final_pool_size"], 30000)
        self.assertEqual(data["planned_top_up_size"], 1000)

    def test_catalog_sources_only_store_prompt_text_and_metadata(self):
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

        for source in data["sources"]:
            self.assertIn("prompt", source["fields_to_keep"])
            self.assertNotIn("image", source["fields_to_keep"])
            self.assertFalse(source["download_images"])


if __name__ == "__main__":
    unittest.main()
