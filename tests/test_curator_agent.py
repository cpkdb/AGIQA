import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data_generation" / "scripts" / "curator_agent.py"


class CuratorAgentTests(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.exists())

    def test_curator_agent_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_root = tmp_path / "runs"
            run_dir = runs_root / "run1"
            run_dir.mkdir(parents=True)

            dataset = {
                "metadata": {"model_id": "flux-schnell"},
                "pairs": [
                    {
                        "id": "pair_0001",
                        "positive": {"prompt": "woman standing in a hallway"},
                        "negative": {"prompt": "woman standing in a hallway with extra arms"},
                        "degradation": {"attribute": "extra_limbs", "severity": "severe"},
                        "validation": {"valid": False, "failure": "insufficient_effect"},
                    },
                    {
                        "id": "pair_0002",
                        "positive": {"prompt": "woman standing in a hallway"},
                        "negative": {"prompt": "woman standing in a hallway with extra arms"},
                        "degradation": {"attribute": "extra_limbs", "severity": "moderate"},
                        "validation": {"valid": False, "failure": "insufficient_effect"},
                    },
                    {
                        "id": "pair_0003",
                        "positive": {"prompt": "red ceramic mug on a desk"},
                        "negative": {"prompt": "red ceramic mug with warped handle"},
                        "degradation": {"attribute": "object_structure_error", "severity": "severe"},
                        "validation": {"valid": True, "failure": None},
                    },
                    {
                        "id": "pair_0004",
                        "positive": {"prompt": "dog beside a bicycle on the street"},
                        "negative": {"prompt": "dog beside a bicycle with overlap issue"},
                        "degradation": {"attribute": "penetration_overlap", "severity": "moderate"},
                        "validation": {"valid": False, "failure": "content_drift"},
                    },
                ],
            }
            full_log = [
                {
                    "pair_id": "pair_0001",
                    "model_id": "flux-schnell",
                    "expected_attribute": "extra_limbs",
                    "positive_prompt": "woman standing in a hallway",
                    "judge_result": {"valid": False, "failure": "insufficient_effect"},
                },
                {
                    "pair_id": "pair_0002",
                    "model_id": "flux-schnell",
                    "expected_attribute": "extra_limbs",
                    "positive_prompt": "woman standing in a hallway",
                    "judge_result": {"valid": False, "failure": "insufficient_effect"},
                },
                {
                    "pair_id": "pair_0003",
                    "model_id": "flux-schnell",
                    "expected_attribute": "object_structure_error",
                    "positive_prompt": "red ceramic mug on a desk",
                    "judge_result": {"valid": True, "failure": None},
                },
                {
                    "pair_id": "pair_0004",
                    "model_id": "flux-schnell",
                    "expected_attribute": "penetration_overlap",
                    "positive_prompt": "dog beside a bicycle on the street",
                    "judge_result": {"valid": False, "failure": "content_drift"},
                },
            ]
            validation_report = {
                "summary": {"total_pairs": 4, "valid_pairs": 1, "invalid_pairs": 3},
                "failure_types": {"insufficient_effect": 2, "content_drift": 1},
            }

            (run_dir / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
            (run_dir / "full_log.json").write_text(json.dumps(full_log, ensure_ascii=False), encoding="utf-8")
            (run_dir / "validation_report.json").write_text(
                json.dumps(validation_report, ensure_ascii=False), encoding="utf-8"
            )

            out_dir = tmp_path / "out"
            subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--runs_root",
                    str(runs_root),
                    "--output_dir",
                    str(out_dir),
                    "--failure_threshold",
                    "2",
                ],
                cwd=str(ROOT),
                check=True,
            )

            expected = [
                "curation_decisions.jsonl",
                "blacklist.json",
                "memory_stats.json",
                "failure_pattern_summary.json",
                "failure_pattern_by_prompt.json",
                "manifest.json",
            ]
            for name in expected:
                self.assertTrue((out_dir / name).exists(), name)

            blacklist = json.loads((out_dir / "blacklist.json").read_text(encoding="utf-8"))
            self.assertEqual(len(blacklist["candidates"]), 1)
            self.assertEqual(blacklist["candidates"][0]["dimension"], "extra_limbs")
            self.assertEqual(blacklist["candidates"][0]["failure_count"], 2)

            memory = json.loads((out_dir / "memory_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(memory["runs"], 1)
            self.assertEqual(memory["pairs"], 4)
            self.assertEqual(memory["invalid_pairs"], 3)
            self.assertEqual(memory["blacklist_candidates"], 1)

            failure = json.loads((out_dir / "failure_pattern_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["totals"]["insufficient_effect"], 2)
            self.assertEqual(failure["totals"]["content_drift"], 1)
            failure_by_prompt = json.loads((out_dir / "failure_pattern_by_prompt.json").read_text(encoding="utf-8"))
            self.assertEqual(failure_by_prompt["rows"][0]["prompt_text"], "woman standing in a hallway")
            self.assertEqual(failure_by_prompt["rows"][0]["dimension"], "extra_limbs")
            self.assertEqual(failure_by_prompt["rows"][0]["failure_count"], 2)

            decisions = [
                json.loads(line)
                for line in (out_dir / "curation_decisions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(decisions), 4)
            self.assertIn("keep", {row["decision"] for row in decisions})
            self.assertIn("review", {row["decision"] for row in decisions})
            self.assertIn("blacklist_candidate", {row["decision"] for row in decisions})

    def test_curator_agent_derives_invalid_pairs_from_full_log_even_if_dataset_only_has_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_root = tmp_path / "runs"
            run_dir = runs_root / "run2"
            run_dir.mkdir(parents=True)

            dataset = {
                "metadata": {"model_id": "sd3.5-large-turbo"},
                "pairs": [
                    {
                        "id": "pair_0000",
                        "positive": {"prompt": "packaging box on a table"},
                        "negative": {"prompt": "packaging box made of jelly"},
                        "degradation": {"attribute": "material_mismatch", "severity": "moderate"},
                        "validation": {"valid": True, "failure": None},
                    }
                ],
            }
            full_log = [
                {
                    "pair_id": "pair_0000",
                    "positive_prompt": "packaging box on a table",
                    "dimension": "material_mismatch",
                    "severity": "moderate",
                    "success": True,
                    "final_validation": {"valid": True, "failure": None},
                    "attempts": [
                        {
                            "status": "success",
                            "validation": {"valid": True, "failure": None},
                        }
                    ],
                },
                {
                    "pair_id": "pair_0001",
                    "positive_prompt": "woman standing in hallway",
                    "dimension": "extra_limbs",
                    "severity": "severe",
                    "success": False,
                    "final_validation": {"valid": False, "failure": "insufficient_effect"},
                    "attempts": [
                        {
                            "status": "failed",
                            "validation": {"valid": False, "failure": "insufficient_effect"},
                        },
                        {
                            "status": "failed",
                            "validation": {"valid": False, "failure": "insufficient_effect"},
                        },
                    ],
                },
                {
                    "pair_id": "pair_0002",
                    "positive_prompt": "woman standing in hallway",
                    "dimension": "extra_limbs",
                    "severity": "moderate",
                    "success": False,
                    "final_validation": {"valid": False, "failure": "insufficient_effect"},
                    "attempts": [
                        {
                            "status": "failed",
                            "validation": {"valid": False, "failure": "insufficient_effect"},
                        }
                    ],
                },
            ]
            validation_report = {
                "summary": {"total_pairs": 3, "valid_pairs": 1, "invalid_pairs": 2},
                "failure_types": {"insufficient_effect": 2},
            }

            (run_dir / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
            (run_dir / "full_log.json").write_text(json.dumps(full_log, ensure_ascii=False), encoding="utf-8")
            (run_dir / "validation_report.json").write_text(
                json.dumps(validation_report, ensure_ascii=False), encoding="utf-8"
            )

            out_dir = tmp_path / "out"
            subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--runs_root",
                    str(runs_root),
                    "--output_dir",
                    str(out_dir),
                    "--failure_threshold",
                    "2",
                ],
                cwd=str(ROOT),
                check=True,
            )

            memory = json.loads((out_dir / "memory_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(memory["pairs"], 3)
            self.assertEqual(memory["invalid_pairs"], 2)
            self.assertEqual(memory["blacklist_candidates"], 1)

            blacklist = json.loads((out_dir / "blacklist.json").read_text(encoding="utf-8"))
            self.assertEqual(len(blacklist["candidates"]), 1)
            self.assertEqual(blacklist["candidates"][0]["prompt_text"], "woman standing in hallway")

            failure_by_prompt = json.loads((out_dir / "failure_pattern_by_prompt.json").read_text(encoding="utf-8"))
            self.assertEqual(failure_by_prompt["rows"][0]["prompt_text"], "woman standing in hallway")
            self.assertEqual(failure_by_prompt["rows"][0]["failure_count"], 2)

            decisions = [
                json.loads(line)
                for line in (out_dir / "curation_decisions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_pair = {row["pair_id"]: row for row in decisions}
            self.assertEqual(by_pair["pair_0000"]["decision"], "keep")
            self.assertEqual(by_pair["pair_0001"]["decision"], "blacklist_candidate")
            self.assertEqual(by_pair["pair_0002"]["decision"], "blacklist_candidate")


if __name__ == "__main__":
    unittest.main()
