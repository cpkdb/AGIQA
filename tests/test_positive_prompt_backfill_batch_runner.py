from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH_RUNNER_PATH = ROOT / "data_generation" / "scripts" / "positive_prompt_backfill_batch_runner.py"


def load_module():
    spec = spec_from_file_location("positive_prompt_backfill_batch_runner", BATCH_RUNNER_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PositivePromptBackfillBatchRunnerTests(unittest.TestCase):
    def test_batch_runner_script_exists(self):
        self.assertTrue(BATCH_RUNNER_PATH.exists())

    def test_build_batched_execution_plan_splits_request_jobs(self):
        module = load_module()

        plan = module.build_batched_execution_plan(
            dimensions=["text_error"],
            batch_size_prompts_per_request=500,
            requests_per_batch=2,
        )

        self.assertEqual(plan["total_requests"], 7)
        self.assertEqual(plan["batch_count"], 4)
        self.assertEqual(plan["batches"][0]["batch_index"], 1)
        self.assertEqual(plan["batches"][0]["request_count"], 2)
        self.assertEqual(plan["batches"][0]["request_ids"], [1, 2])
        self.assertEqual(plan["batches"][3]["request_ids"], [7])

    def test_write_batched_execution_plan_writes_index_and_batch_manifests(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = module.write_batched_execution_plan(
                output_dir=tmpdir,
                dimensions=["text_error"],
                batch_size_prompts_per_request=1000,
                requests_per_batch=3,
            )

            index_path = Path(result["batch_index_path"])
            self.assertTrue(index_path.exists())

            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total_requests"], 6)
            self.assertEqual(payload["batch_count"], 2)

            batch_one_manifest = Path(payload["batches"][0]["manifest_path"])
            batch_two_manifest = Path(payload["batches"][1]["manifest_path"])
            self.assertTrue(batch_one_manifest.exists())
            self.assertTrue(batch_two_manifest.exists())

    def test_get_batch_status_marks_completed_and_pending_batches(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            module.write_batched_execution_plan(
                output_dir=tmpdir,
                dimensions=["text_error"],
                batch_size_prompts_per_request=1000,
                requests_per_batch=3,
            )

            batch_one_dir = Path(tmpdir) / "batches" / "batch_0001"
            (batch_one_dir / "generated_positive_prompts.jsonl").write_text("", encoding="utf-8")
            (batch_one_dir / "execution_summary.json").write_text(
                json.dumps({"total_generated_prompts": 3}),
                encoding="utf-8",
            )

            status = module.get_batch_status(tmpdir)

            self.assertEqual(status["batch_count"], 2)
            self.assertEqual(status["completed_batches"], 1)
            self.assertEqual(status["next_pending_batch_index"], 2)
            self.assertEqual(status["batches"][0]["status"], "completed")
            self.assertEqual(status["batches"][1]["status"], "pending")

    def test_execute_next_pending_batch_runs_first_pending_batch(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            module.write_batched_execution_plan(
                output_dir=tmpdir,
                dimensions=["text_error"],
                batch_size_prompts_per_request=1000,
                requests_per_batch=3,
            )

            batch_one_dir = Path(tmpdir) / "batches" / "batch_0001"
            (batch_one_dir / "generated_positive_prompts.jsonl").write_text("", encoding="utf-8")
            (batch_one_dir / "execution_summary.json").write_text(
                json.dumps({"total_generated_prompts": 3}),
                encoding="utf-8",
            )

            captured = {}

            class FakeExecutor:
                @staticmethod
                def execute_request_plan(request_plan, llm_config_path, output_dir, sleep_seconds=0.0):
                    captured["request_plan"] = request_plan
                    captured["llm_config_path"] = llm_config_path
                    captured["output_dir"] = output_dir
                    captured["sleep_seconds"] = sleep_seconds
                    return {"generated_path": str(Path(output_dir) / "generated_positive_prompts.jsonl")}

            with patch.object(module, "_load_executor_module", return_value=FakeExecutor):
                result = module.execute_next_pending_batch(
                    output_dir=tmpdir,
                    llm_config_path="/tmp/fake_llm.yaml",
                    sleep_seconds=0.5,
                )

            self.assertEqual(result["executed_batch_index"], 2)
            self.assertEqual(Path(captured["output_dir"]).name, "batch_0002")
            self.assertEqual(captured["llm_config_path"], "/tmp/fake_llm.yaml")
            self.assertEqual(captured["sleep_seconds"], 0.5)

    def test_ensure_batch_plan_reuses_existing_index_without_rewriting(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            first = module.write_batched_execution_plan(
                output_dir=tmpdir,
                dimensions=["text_error"],
                batch_size_prompts_per_request=1000,
                requests_per_batch=3,
            )
            first_payload = json.loads(Path(first["batch_index_path"]).read_text(encoding="utf-8"))

            second = module.ensure_batched_execution_plan(
                output_dir=tmpdir,
                dimensions=["hand_malformation"],
                batch_size_prompts_per_request=5,
                requests_per_batch=1,
            )
            second_payload = json.loads(Path(second["batch_index_path"]).read_text(encoding="utf-8"))

            self.assertEqual(first_payload["total_requests"], second_payload["total_requests"])
            self.assertEqual(first_payload["batch_count"], second_payload["batch_count"])

    def test_execute_all_pending_batches_runs_until_complete(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            module.write_batched_execution_plan(
                output_dir=tmpdir,
                dimensions=["text_error"],
                batch_size_prompts_per_request=1000,
                requests_per_batch=3,
            )

            executed_dirs = []

            class FakeExecutor:
                @staticmethod
                def execute_request_plan(request_plan, llm_config_path, output_dir, sleep_seconds=0.0):
                    out = Path(output_dir)
                    executed_dirs.append(out.name)
                    (out / "generated_positive_prompts.jsonl").write_text("", encoding="utf-8")
                    (out / "execution_summary.json").write_text(
                        json.dumps({"total_generated_prompts": request_plan["total_target_prompts"]}),
                        encoding="utf-8",
                    )
                    return {"generated_path": str(out / "generated_positive_prompts.jsonl")}

            with patch.object(module, "_load_executor_module", return_value=FakeExecutor):
                result = module.execute_all_pending_batches(
                    output_dir=tmpdir,
                    llm_config_path="/tmp/fake_llm.yaml",
                    sleep_seconds=0.1,
                )

            self.assertEqual(result["executed_batch_count"], 2)
            self.assertEqual(result["completed_batches"], 2)
            self.assertEqual(result["pending_batches"], 0)
            self.assertEqual(executed_dirs, ["batch_0001", "batch_0002"])


if __name__ == "__main__":
    unittest.main()
