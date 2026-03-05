from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER_PATH = ROOT / "data_generation" / "scripts" / "prompt_source_downloader.py"


def load_module():
    spec = spec_from_file_location("prompt_source_downloader", DOWNLOADER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptSourceDownloaderTests(unittest.TestCase):
    def test_prompt_source_downloader_script_exists(self):
        self.assertTrue(DOWNLOADER_PATH.exists())

    def test_build_download_manifest_returns_metadata_only_jobs(self):
        module = load_module()

        manifest = module.build_download_manifest()

        self.assertEqual(manifest["mode"], "metadata_only")
        self.assertEqual(len(manifest["jobs"]), 5)

        first_job = manifest["jobs"][0]
        self.assertEqual(first_job["source"], "diffusiondb")
        self.assertEqual(first_job["candidate_pool_size"], 60000)
        self.assertFalse(first_job["download_images"])
        self.assertIn("prompt", first_job["fields_to_keep"])

    def test_write_download_manifest_creates_json_file(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = module.write_download_manifest(tmpdir)
            manifest_file = Path(manifest_path)

            self.assertTrue(manifest_file.exists())

            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(data["mode"], "metadata_only")
            self.assertEqual(len(data["jobs"]), 5)
            self.assertEqual(data["jobs"][1]["source"], "pick_a_pic_v2")


if __name__ == "__main__":
    unittest.main()
