from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data_generation" / "scripts" / "prompt_source_plan.py"


def load_module():
    spec = spec_from_file_location("prompt_source_plan", PLAN_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptSourcePlanTests(unittest.TestCase):
    def test_prompt_source_plan_script_exists(self):
        self.assertTrue(PLAN_PATH.exists())

    def test_build_download_plan_validates_catalog_and_sampling_spec(self):
        module = load_module()

        plan = module.build_download_plan()

        self.assertEqual(plan["target_final_pool_size"], 30000)
        self.assertEqual(plan["planned_selected_before_top_up"], 29000)
        self.assertEqual(plan["planned_top_up_size"], 1000)
        self.assertEqual(
            [source["name"] for source in plan["sources"]],
            [
                "diffusiondb",
                "pick_a_pic_v2",
                "t2i_compbench",
                "parti_prompts",
                "longbench_t2i",
            ],
        )

        diffusiondb = plan["source_index"]["diffusiondb"]
        self.assertEqual(diffusiondb["candidate_pool_size"], 60000)
        self.assertEqual(diffusiondb["final_selected_size"], 14000)
        self.assertEqual(len(diffusiondb["buckets"]), 7)

    def test_prepare_local_workspace_creates_source_and_bucket_dirs(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = module.prepare_local_workspace(tmpdir)

            self.assertTrue((Path(tmpdir) / "diffusiondb").exists())
            self.assertTrue((Path(tmpdir) / "diffusiondb" / "people_face_hands").exists())
            self.assertTrue((Path(tmpdir) / "pick_a_pic_v2" / "people_and_activities").exists())
            self.assertTrue((Path(tmpdir) / "t2i_compbench").exists())
            self.assertEqual(result["workspace_dir"], str(Path(tmpdir).resolve()))


if __name__ == "__main__":
    unittest.main()
