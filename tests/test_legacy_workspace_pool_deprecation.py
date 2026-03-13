from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "data_generation" / "scripts" / "run_diagnostic.sh",
    ROOT / "data_generation" / "scripts" / "run_pipeline_hunyuan.sh",
    ROOT / "data_generation" / "scripts" / "run_pipeline_flux.sh",
    ROOT / "data_generation" / "scripts" / "run_demo_v3_dimension_paired.sh",
    ROOT / "data_generation" / "scripts" / "run_demo_v3_dimension_paired_flux.sh",
    ROOT / "data_generation" / "scripts" / "run_demo_v3_dimension_paired_flux_schnell.sh",
    ROOT / "data_generation" / "scripts" / "demo_v3_dimension_paired.py",
]


class LegacyWorkspacePoolDeprecationTests(unittest.TestCase):
    def test_runtime_and_demo_scripts_no_longer_reference_working_pool_v1(self):
        for path in SCRIPTS:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("/prompt_sources_workspace/working_pool_v1.jsonl", source, path.name)
            self.assertIn("merged_working_pool_cleaned_v1.jsonl", source, path.name)


if __name__ == "__main__":
    unittest.main()
