from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "data_generation" / "scripts" / "semantic_screened_pool_finalizer.py"


def load_module():
    spec = spec_from_file_location("semantic_screened_pool_finalizer", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SemanticScreenedPoolFinalizerTests(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists())

    def test_finalize_common_outputs_uses_unified_directory_names(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs_root = root / "runs"
            out_root = root / "out"
            runs_root.mkdir()

            family_runs = {
                "human_full_body_realistic": [
                    {"family_name": "human_full_body_realistic", "sample_id": "h1", "prompt": "woman standing", "label": "pass"},
                    {"family_name": "human_full_body_realistic", "sample_id": "h2", "prompt": "portrait closeup", "label": "fail"},
                ],
                "structured_object_primary": [
                    {"family_name": "structured_object_primary", "sample_id": "o1", "prompt": "ceramic mug on desk", "label": "pass"},
                ],
                "multi_object_reference": [
                    {"family_name": "multi_object_reference", "sample_id": "m1", "prompt": "mug beside laptop", "label": "pass"},
                    {"family_name": "multi_object_reference", "sample_id": "m2", "prompt": "dog beside bowl", "label": "pass"},
                ],
            }
            run_dir_map = {}
            for family, rows in family_runs.items():
                run_dir = runs_root / family
                run_dir.mkdir()
                results = run_dir / "family_screening_results.jsonl"
                results.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
                run_dir_map[family] = str(run_dir)

            screening_spec = root / "prompt_pool_screening_spec_v1.json"
            screening_spec.write_text(
                json.dumps(
                    {
                        "shared_pool_families": {
                            "human_full_body_realistic": {},
                            "structured_object_primary": {},
                            "multi_object_reference": {},
                        },
                        "dimension_overrides": {
                            "body_proportion_error": {"shared_pool_family": "human_full_body_realistic"},
                            "extra_limbs": {"shared_pool_family": "human_full_body_realistic"},
                            "object_structure_error": {"shared_pool_family": "structured_object_primary"},
                            "material_mismatch": {"shared_pool_family": "structured_object_primary"},
                            "scale_inconsistency": {"shared_pool_family": "multi_object_reference"},
                            "penetration_overlap": {"shared_pool_family": "multi_object_reference"},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            build_targets = root / "prompt_pool_build_targets_v1.json"
            build_targets.write_text(
                json.dumps(
                    {
                        "shared_pool_family_outputs": {
                            "common": {
                                "directory_name": "shared_family_screened_pools_cleaned_v1",
                                "index_name": "index.json",
                            },
                            "turbo": {
                                "directory_name": "sd35_turbo_shared_family_screened_pools_clipsafe_v1",
                                "index_name": "index.json",
                            },
                        },
                        "dimension_subpool_outputs": {
                            "common": {
                                "directory_name": "semantic_screened_dimension_subpools_cleaned_v1",
                                "index_name": "index.json",
                            },
                            "turbo": {
                                "directory_name": "sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1",
                                "index_name": "index.json",
                            },
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            outputs = module.finalize_common_screened_pools(
                family_run_dirs=run_dir_map,
                output_root=str(out_root),
                screening_spec_path=str(screening_spec),
                build_targets_path=str(build_targets),
            )

            family_dir = Path(outputs["common_family_dir"])
            dimension_dir = Path(outputs["common_dimension_dir"])
            self.assertEqual(family_dir.name, "shared_family_screened_pools_cleaned_v1")
            self.assertEqual(dimension_dir.name, "semantic_screened_dimension_subpools_cleaned_v1")

            human_rows = [
                json.loads(line)
                for line in (family_dir / "human_full_body_realistic.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(human_rows), 1)
            self.assertEqual(human_rows[0]["prompt"], "woman standing")

            body_rows = [
                json.loads(line)
                for line in (dimension_dir / "body_proportion_error.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(body_rows), 1)
            self.assertEqual(body_rows[0]["prompt"], "woman standing")

            index_payload = json.loads((dimension_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index_payload["dimensions"]["scale_inconsistency"]["count"], 2)
            self.assertEqual(index_payload["dimensions"]["scale_inconsistency"]["source_family"], "multi_object_reference")

    def test_finalize_common_outputs_accepts_sharded_family_run_root(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs_root = root / "runs"
            out_root = root / "out"
            runs_root.mkdir()

            human_run = runs_root / "human"
            human_run.mkdir()
            (human_run / "family_screening_results.jsonl").write_text(
                json.dumps({"family_name": "human_full_body_realistic", "sample_id": "h1", "prompt": "woman standing", "label": "pass"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            object_root = runs_root / "object_root"
            (object_root / "shard_0").mkdir(parents=True)
            (object_root / "shard_1").mkdir(parents=True)
            (object_root / "shard_0" / "family_screening_results.jsonl").write_text(
                json.dumps({"family_name": "structured_object_primary", "sample_id": "o1", "prompt": "mug on desk", "label": "pass"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            (object_root / "shard_1" / "family_screening_results.jsonl").write_text(
                json.dumps({"family_name": "structured_object_primary", "sample_id": "o2", "prompt": "laptop on table", "label": "pass"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            multi_run = runs_root / "multi"
            multi_run.mkdir()
            (multi_run / "family_screening_results.jsonl").write_text(
                json.dumps({"family_name": "multi_object_reference", "sample_id": "m1", "prompt": "mug beside laptop", "label": "pass"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            screening_spec = root / "prompt_pool_screening_spec_v1.json"
            screening_spec.write_text(
                json.dumps(
                    {
                        "shared_pool_families": {
                            "human_full_body_realistic": {},
                            "structured_object_primary": {},
                            "multi_object_reference": {},
                        },
                        "dimension_overrides": {
                            "body_proportion_error": {"shared_pool_family": "human_full_body_realistic"},
                            "object_structure_error": {"shared_pool_family": "structured_object_primary"},
                            "scale_inconsistency": {"shared_pool_family": "multi_object_reference"},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            build_targets = root / "prompt_pool_build_targets_v1.json"
            build_targets.write_text(
                json.dumps(
                    {
                        "shared_pool_family_outputs": {
                            "common": {
                                "directory_name": "shared_family_screened_pools_cleaned_v1",
                                "index_name": "index.json",
                            }
                        },
                        "dimension_subpool_outputs": {
                            "common": {
                                "directory_name": "semantic_screened_dimension_subpools_cleaned_v1",
                                "index_name": "index.json",
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            outputs = module.finalize_common_screened_pools(
                family_run_dirs={
                    "human_full_body_realistic": str(human_run),
                    "structured_object_primary": str(object_root),
                    "multi_object_reference": str(multi_run),
                },
                output_root=str(out_root),
                screening_spec_path=str(screening_spec),
                build_targets_path=str(build_targets),
            )

            object_rows = [
                json.loads(line)
                for line in (Path(outputs["common_family_dir"]) / "structured_object_primary.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(object_rows), 2)

    def test_finalize_turbo_outputs_reuses_common_passes_then_filters_by_tokens(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common_family_dir = root / "shared_family_screened_pools_cleaned_v1"
            common_dimension_dir = root / "semantic_screened_dimension_subpools_cleaned_v1"
            common_family_dir.mkdir()
            common_dimension_dir.mkdir()

            family_index = {
                "families": {
                    "human_full_body_realistic": {
                        "filename": str((common_family_dir / "human_full_body_realistic.jsonl").resolve()),
                        "target_dimensions": ["body_proportion_error", "extra_limbs"],
                    }
                }
            }
            (common_family_dir / "index.json").write_text(json.dumps(family_index, ensure_ascii=False, indent=2), encoding="utf-8")
            (common_family_dir / "human_full_body_realistic.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "short prompt", "family_name": "human_full_body_realistic"}, ensure_ascii=False),
                        json.dumps({"prompt": "this is a much longer prompt that should be dropped", "family_name": "human_full_body_realistic"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            dimension_index = {
                "dimensions": {
                    "body_proportion_error": {
                        "filename": str((common_dimension_dir / "body_proportion_error.jsonl").resolve()),
                        "source_family": "human_full_body_realistic",
                    }
                }
            }
            (common_dimension_dir / "index.json").write_text(json.dumps(dimension_index, ensure_ascii=False, indent=2), encoding="utf-8")
            (common_dimension_dir / "body_proportion_error.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "short prompt", "family_name": "human_full_body_realistic"}, ensure_ascii=False),
                        json.dumps({"prompt": "this is a much longer prompt that should be dropped", "family_name": "human_full_body_realistic"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            build_targets = root / "prompt_pool_build_targets_v1.json"
            build_targets.write_text(
                json.dumps(
                    {
                        "shared_pool_family_outputs": {
                            "turbo": {
                                "directory_name": "sd35_turbo_shared_family_screened_pools_clipsafe_v1",
                                "index_name": "index.json",
                            }
                        },
                        "dimension_subpool_outputs": {
                            "turbo": {
                                "directory_name": "sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1",
                                "index_name": "index.json",
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            outputs = module.finalize_turbo_derived_screened_pools(
                common_family_dir=str(common_family_dir),
                common_dimension_dir=str(common_dimension_dir),
                output_root=str(root / "out"),
                build_targets_path=str(build_targets),
                measure_tokens=lambda prompt: len(prompt.split()),
                global_max_tokens=4,
                strict_max_tokens=3,
            )

            turbo_family_rows = [
                json.loads(line)
                for line in (Path(outputs["turbo_family_dir"]) / "human_full_body_realistic.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            turbo_dimension_rows = [
                json.loads(line)
                for line in (Path(outputs["turbo_dimension_dir"]) / "body_proportion_error.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(len(turbo_family_rows), 1)
            self.assertEqual(len(turbo_dimension_rows), 1)
            self.assertEqual(turbo_family_rows[0]["prompt"], "short prompt")
            self.assertEqual(turbo_dimension_rows[0]["prompt"], "short prompt")

    def test_finalize_common_outputs_adds_rule_only_floating_objects_from_source_pool(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs_root = root / "runs"
            out_root = root / "out"
            runs_root.mkdir()

            for family in ["human_full_body_realistic", "structured_object_primary", "multi_object_reference"]:
                run_dir = runs_root / family
                run_dir.mkdir()
                (run_dir / "family_screening_results.jsonl").write_text("", encoding="utf-8")

            source_prompts = root / "merged_working_pool_cleaned_v1.jsonl"
            source_rows = [
                {"prompt": "a ceramic mug on a desk", "semantic_tags": ["has_structured_object"], "signature": {"has_structured_object": True}},
                {"prompt": "a woman walking in a park", "semantic_tags": ["has_person"], "signature": {"has_person": True}},
                {"prompt": "a bird on a branch", "semantic_tags": ["has_animal"], "signature": {"has_animal": True}},
                {"prompt": "abstract color field", "semantic_tags": [], "signature": {}},
            ]
            source_prompts.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
                encoding="utf-8",
            )

            screening_spec = root / "prompt_pool_screening_spec_v1.json"
            screening_spec.write_text(
                json.dumps(
                    {
                        "shared_pool_families": {
                            "human_full_body_realistic": {},
                            "structured_object_primary": {},
                            "multi_object_reference": {},
                        },
                        "dimension_overrides": {
                            "floating_objects": {
                                "shared_pool_family": None,
                                "requires_llm_screen": False,
                                "builder_mode": "rule_recall_only",
                                "base_tags": [
                                    "has_structured_object",
                                    "has_countable_objects",
                                    "has_person",
                                    "has_animal",
                                ],
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            build_targets = root / "prompt_pool_build_targets_v1.json"
            build_targets.write_text(
                json.dumps(
                    {
                        "shared_pool_family_outputs": {
                            "common": {
                                "directory_name": "shared_family_screened_pools_cleaned_v1",
                                "index_name": "index.json",
                            }
                        },
                        "dimension_subpool_outputs": {
                            "common": {
                                "directory_name": "semantic_screened_dimension_subpools_cleaned_v1",
                                "index_name": "index.json",
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            outputs = module.finalize_common_screened_pools(
                family_run_dirs={
                    "human_full_body_realistic": str(runs_root / "human_full_body_realistic"),
                    "structured_object_primary": str(runs_root / "structured_object_primary"),
                    "multi_object_reference": str(runs_root / "multi_object_reference"),
                },
                output_root=str(out_root),
                screening_spec_path=str(screening_spec),
                build_targets_path=str(build_targets),
                common_source_prompts_path=str(source_prompts),
            )

            floating_rows = [
                json.loads(line)
                for line in (Path(outputs["common_dimension_dir"]) / "floating_objects.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["prompt"] for row in floating_rows], [
                "a ceramic mug on a desk",
                "a woman walking in a park",
                "a bird on a branch",
            ])

            index_payload = json.loads((Path(outputs["common_dimension_dir"]) / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index_payload["dimensions"]["floating_objects"]["count"], 3)
            self.assertEqual(index_payload["dimensions"]["floating_objects"]["source_family"], "rule_recall_only")


if __name__ == "__main__":
    unittest.main()
