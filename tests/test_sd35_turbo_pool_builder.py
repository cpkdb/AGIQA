from pathlib import Path
import importlib.util
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "data_generation" / "scripts" / "sd35_turbo_pool_builder.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sd35_turbo_pool_builder_under_test", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SD35TurboPoolBuilderTests(unittest.TestCase):
    def test_dimension_specific_thresholds_apply(self):
        module = load_module()
        strict_dims = {"face_asymmetry"}
        records = [
            {"prompt": "keep-short"},
            {"prompt": "drop-strict-only"},
            {"prompt": "drop-global"},
        ]
        token_lengths = {
            "keep-short": 35,
            "drop-strict-only": 45,
            "drop-global": 55,
        }

        def fake_measure(prompt):
            return token_lengths[prompt]

        cleaned_regular, summary_regular = module.filter_records_by_clip_tokens(
            records,
            max_tokens=50,
            measure_tokens=fake_measure,
        )
        cleaned_strict, summary_strict = module.filter_records_by_clip_tokens(
            records,
            max_tokens=40,
            measure_tokens=fake_measure,
        )

        self.assertEqual([r["prompt"] for r in cleaned_regular], ["keep-short", "drop-strict-only"])
        self.assertEqual(summary_regular["dropped_too_long"], 1)
        self.assertEqual([r["prompt"] for r in cleaned_strict], ["keep-short"])
        self.assertEqual(summary_strict["dropped_too_long"], 2)
        self.assertIn("face_asymmetry", strict_dims)


if __name__ == "__main__":
    unittest.main()
