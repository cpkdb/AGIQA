import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data_generation" / "scripts" / "orchestrator.py"
TOOLS = ROOT / "data_generation" / "scripts" / "orchestrator_tools.py"


def _load_tools():
    for path in [str(ROOT), str(ROOT / "data_generation" / "scripts")]:
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("orchestrator_tools", TOOLS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OrchestratorTests(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.exists())

    def test_orchestrator_writes_minimal_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "orch"
            subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--output_dir",
                    str(out_dir),
                    "--model_id",
                    "flux-schnell",
                    "--dry_run",
                ],
                cwd=str(ROOT),
                check=True,
            )

            for name in ["run_config.json", "run_registry.json", "launch_command.sh"]:
                self.assertTrue((out_dir / name).exists(), name)

            run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["model_id"], "flux-schnell")
            self.assertIn("source_prompts", run_config)
            self.assertIn("dimension_subpool_index", run_config)
            self.assertIn("steps", run_config)
            self.assertIn("cfg", run_config)
            self.assertIn("output_dir", run_config)

            registry = json.loads((out_dir / "run_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["status"], "planned")
            self.assertEqual(registry["model_id"], "flux-schnell")
            self.assertEqual(registry["run_config_path"], str((out_dir / "run_config.json").resolve()))

    def test_execute_launch_script_writes_log_and_returns_code(self):
        tools = _load_tools()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "run.sh"
            script.write_text("#!/bin/bash\necho orchestrator-ok\n", encoding="utf-8")
            script.chmod(0o755)
            log_path = tmp_path / "run.log"

            return_code = tools.execute_launch_script(script, log_path)

            self.assertEqual(return_code, 0)
            self.assertIn("orchestrator-ok", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
