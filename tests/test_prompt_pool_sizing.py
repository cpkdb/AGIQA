from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SIZING_PATH = ROOT / "data_generation" / "scripts" / "prompt_pool_sizing.py"


def load_module():
    spec = spec_from_file_location("prompt_pool_sizing", SIZING_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptPoolSizingTests(unittest.TestCase):
    def test_prompt_pool_sizing_script_exists(self):
        self.assertTrue(SIZING_PATH.exists())

    def test_estimate_prompt_pool_size_returns_dimension_and_total_targets(self):
        module = load_module()

        result = module.estimate_prompt_pool_size(
            total_pairs=1000,
            num_negatives_per_positive=5,
            dimension_weights={
                "hand_malformation": 0.2,
                "blur": 0.8,
            },
            constrained_dimensions={"hand_malformation"},
            constrained_multiplier=1.5,
            min_per_dimension=10,
        )

        self.assertEqual(result["target_positive_pool_size"], 200)
        self.assertEqual(result["total_pairs"], 1000)
        self.assertEqual(result["num_negatives_per_positive"], 5)
        self.assertEqual(
            result["per_dimension"]["blur"]["estimated_pairs"],
            800,
        )
        self.assertEqual(
            result["per_dimension"]["blur"]["required_positive_prompts"],
            160,
        )
        self.assertEqual(
            result["per_dimension"]["hand_malformation"]["estimated_pairs"],
            200,
        )
        self.assertEqual(
            result["per_dimension"]["hand_malformation"]["required_positive_prompts"],
            60,
        )
        self.assertTrue(result["per_dimension"]["hand_malformation"]["is_constrained"])

    def test_rank_prompt_sources_prefers_higher_coverage_and_volume(self):
        module = load_module()

        ranked = module.rank_prompt_sources(
            [
                {
                    "name": "small_high_fit",
                    "prompt_count": 200,
                    "dimension_coverage": {"hand_malformation": 0.8, "blur": 0.7},
                },
                {
                    "name": "large_medium_fit",
                    "prompt_count": 1000,
                    "dimension_coverage": {"hand_malformation": 0.5, "blur": 0.5},
                },
                {
                    "name": "tiny_low_fit",
                    "prompt_count": 50,
                    "dimension_coverage": {"hand_malformation": 0.2, "blur": 0.1},
                },
            ],
            target_dimensions=["hand_malformation", "blur"],
        )

        self.assertEqual(
            [item["name"] for item in ranked],
            ["large_medium_fit", "small_high_fit", "tiny_low_fit"],
        )
        self.assertGreater(ranked[0]["selection_score"], ranked[1]["selection_score"])
        self.assertGreater(ranked[1]["selection_score"], ranked[2]["selection_score"])


if __name__ == "__main__":
    unittest.main()
