from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = ROOT / "data_generation" / "scripts" / "prompt_pool_agent.py"


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("prompt_pool_agent", AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptPoolAgentTests(unittest.TestCase):
    def test_agent_writes_manifest_inventory_routing_screening_and_cleanup_artifacts(self):
        agent = _load_agent_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "agent_output"

            cleaned_source = root / "merged_working_pool_cleaned_v1.jsonl"
            cleaned_source.write_text(
                json.dumps({"prompt": "a"}) + "\n" + json.dumps({"prompt": "b"}) + "\n",
                encoding="utf-8",
            )
            turbo_source = root / "merged_working_pool_sd35_turbo_clipsafe_v1.jsonl"
            turbo_source.write_text(
                json.dumps({"prompt": "t1"}) + "\n",
                encoding="utf-8",
            )

            anatomy_v2 = root / "anatomy_screened_dimension_subpools_cleaned_v2" / "index.json"
            anatomy_v2.parent.mkdir(parents=True, exist_ok=True)
            anatomy_v2.write_text(
                json.dumps(
                    {
                        "dimensions": {
                            "blur": {"filename": "blur.jsonl", "count": 2},
                            "hand_malformation": {"filename": "hand_malformation.jsonl", "count": 1},
                        }
                    }
                ),
                encoding="utf-8",
            )

            turbo_v2 = root / "sd35_turbo_dimension_subpools_clipsafe_v2" / "index.json"
            turbo_v2.parent.mkdir(parents=True, exist_ok=True)
            turbo_v2.write_text(
                json.dumps(
                    {
                        "dimensions": {
                            "blur": {"filename": "blur.jsonl", "count": 1},
                            "hand_malformation": {"filename": "hand_malformation.jsonl", "count": 1},
                        }
                    }
                ),
                encoding="utf-8",
            )

            inventory_root = root / "pool_inventory"
            inventory_root.mkdir(parents=True, exist_ok=True)
            (inventory_root / "targeted_dimension_subpools_cleaned_v1").mkdir()
            (inventory_root / "sd35_turbo_targeted_dimension_subpools_clipsafe_v1").mkdir()
            (inventory_root / "anatomy_screened_dimension_subpools_cleaned_v2").mkdir()
            (inventory_root / "anatomy_screened_dimension_subpools_v2").mkdir()
            (inventory_root / "merged_working_pool_cleaned_v1.jsonl").write_text(
                json.dumps({"prompt": "inventory"}) + "\n",
                encoding="utf-8",
            )

            taxonomy_path = root / "quality_dimensions_active.json"
            taxonomy_path.write_text(
                json.dumps(
                    {
                        "taxonomy_name": "test_taxonomy",
                        "statistics": {"total_dimensions": 4},
                        "perspectives": {
                            "technical_quality": {
                                "dimensions": {
                                    "blur": {"zh": "模糊"},
                                }
                            },
                            "semantic_rationality": {
                                "dimensions": {
                                    "hand_malformation": {"zh": "手部畸形"},
                                    "body_proportion_error": {"zh": "身体比例错误"},
                                    "extra_limbs": {"zh": "额外肢体"},
                                    "object_structure_error": {"zh": "物体结构错误"},
                                    "material_mismatch": {"zh": "材质错配"},
                                    "scale_inconsistency": {"zh": "尺度不一致"},
                                    "floating_objects": {"zh": "漂浮物体"},
                                    "penetration_overlap": {"zh": "穿插重叠"},
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            routing_config_path = root / "prompt_pool_routing_v1.json"
            routing_config_path.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "routing": {
                            "blur": {
                                "strategy": "global_cleaned_pool",
                                "notes": "全池可用",
                            },
                            "hand_malformation": {
                                "strategy": "existing_special_pool",
                                "notes": "沿用现有 anatomy 子池",
                            },
                            "body_proportion_error": {
                                "strategy": "rule_recall_then_llm_screen",
                                "base_tags": ["has_person"],
                                "shared_pool_family": "human_full_body_realistic",
                                "llm_screening_required": True,
                                "turbo_pool_strategy": "reuse_common_semantic_screen_then_token_filter",
                                "notes": "需要单人真实全身人物，头身和腿部比例可读，排除铠甲/重度遮挡/夸张服饰。",
                            },
                            "extra_limbs": {
                                "strategy": "rule_recall_then_llm_screen",
                                "base_tags": ["has_person"],
                                "shared_pool_family": "human_full_body_realistic",
                                "llm_screening_required": True,
                                "turbo_pool_strategy": "reuse_common_semantic_screen_then_token_filter",
                                "notes": "与 body 共用真实全身人物父池，再二次筛选可读肢体。",
                            },
                            "object_structure_error": {
                                "strategy": "rule_recall_then_llm_screen",
                                "base_tags": ["has_structured_object"],
                                "shared_pool_family": "structured_object_primary",
                                "turbo_pool_strategy": "reuse_common_semantic_screen_then_token_filter",
                                "notes": "需要结构化非生物物体",
                            },
                            "material_mismatch": {
                                "strategy": "rule_recall_then_llm_screen",
                                "base_tags": ["has_structured_object"],
                                "shared_pool_family": "structured_object_primary",
                                "llm_screening_required": True,
                                "turbo_pool_strategy": "reuse_common_semantic_screen_then_token_filter",
                                "notes": "与 object 共用结构化物体父池，再筛材质可读性。",
                            },
                            "scale_inconsistency": {
                                "strategy": "rule_recall_then_llm_screen",
                                "base_tags": ["has_multiple_objects"],
                                "shared_pool_family": "multi_object_reference",
                                "turbo_pool_strategy": "reuse_common_semantic_screen_then_token_filter",
                                "notes": "需要多物体参照",
                            },
                            "floating_objects": {
                                "strategy": "rule_recall_then_llm_screen",
                                "base_tags": ["has_structured_object", "has_countable_objects"],
                                "shared_pool_family": "multi_object_reference",
                                "llm_screening_required": True,
                                "turbo_pool_strategy": "reuse_common_rule_pool_then_token_filter",
                                "notes": "共用多物体参照父池，再筛支撑关系。",
                            },
                            "penetration_overlap": {
                                "strategy": "rule_recall_then_llm_screen",
                                "base_tags": ["has_multiple_objects", "has_structured_object"],
                                "shared_pool_family": "multi_object_reference",
                                "llm_screening_required": True,
                                "turbo_pool_strategy": "reuse_common_semantic_screen_then_token_filter",
                                "notes": "共用多物体参照父池，再筛边界和接触关系。",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            candidates = agent.taxonomy_tools.RuntimeResourceCandidates(
                base_source_prompts=root / "merged_working_pool.jsonl",
                cleaned_source_prompts=cleaned_source,
                sd35_turbo_source_prompts=turbo_source,
                base_dimension_subpool_index=root / "dimension_subpools" / "index.json",
                base_cleaned_dimension_subpool_index=root / "dimension_subpools_cleaned_v1" / "index.json",
                semantic_screened_dimension_subpool_index_v1=root / "semantic_screened_dimension_subpools_cleaned_v1" / "index.json",
                screened_cleaned_dimension_subpool_index_v2=anatomy_v2,
                screened_cleaned_dimension_subpool_index_v1=root / "anatomy_screened_dimension_subpools_cleaned_v1" / "index.json",
                screened_dimension_subpool_index=root / "anatomy_screened_dimension_subpools" / "index.json",
                sd35_turbo_semantic_screened_dimension_subpool_index_v1=root / "sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1" / "index.json",
                sd35_turbo_dimension_subpool_index_v2=turbo_v2,
                sd35_turbo_dimension_subpool_index_v1=root / "sd35_turbo_dimension_subpools_clipsafe_v1" / "index.json",
            )

            result = agent.run_prompt_pool_agent(
                output_dir=output_dir,
                taxonomy_path=taxonomy_path,
                resource_candidates=candidates,
                inventory_roots=[inventory_root],
                routing_config_path=routing_config_path,
            )

            expected_files = {
                "active_pool_manifest": output_dir / "active_pool_manifest.json",
                "prompt_pool_inventory": output_dir / "prompt_pool_inventory.json",
                "prompt_pool_routing": output_dir / "prompt_pool_routing_v1.json",
                "prompt_pool_screening_plan": output_dir / "prompt_pool_screening_plan_v1.json",
                "prompt_pool_screening_spec": output_dir / "prompt_pool_screening_spec_v1.json",
                "prompt_pool_build_targets": output_dir / "prompt_pool_build_targets_v1.json",
                "prompt_pool_cleanup_candidates": output_dir / "prompt_pool_cleanup_candidates_v1.json",
                "manifest": output_dir / "manifest.json",
            }
            for file_path in expected_files.values():
                self.assertTrue(file_path.exists(), str(file_path))

            manifest = json.loads(expected_files["active_pool_manifest"].read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["runtime_resources"]["sd3.5-large-turbo"]["source_prompts"],
                str(turbo_source),
            )
            self.assertEqual(
                manifest["coverage_by_model"]["flux-schnell"]["dimension_counts"]["hand_malformation"],
                1,
            )

            inventory = json.loads(expected_files["prompt_pool_inventory"].read_text(encoding="utf-8"))
            self.assertIn("active", inventory["classifications"])
            self.assertIn("stale_candidate", inventory["classifications"])
            active_names = {Path(entry["path"]).name for entry in inventory["classifications"]["active"]}
            stale_names = {Path(entry["path"]).name for entry in inventory["classifications"]["stale_candidate"]}
            historical_names = {Path(entry["path"]).name for entry in inventory["classifications"]["historical_candidate"]}
            self.assertIn("anatomy_screened_dimension_subpools_cleaned_v2", active_names)
            self.assertIn("targeted_dimension_subpools_cleaned_v1", stale_names)
            self.assertIn("anatomy_screened_dimension_subpools_v2", historical_names)

            routing = json.loads(expected_files["prompt_pool_routing"].read_text(encoding="utf-8"))
            self.assertEqual(routing["dimensions"]["blur"]["strategy"], "global_cleaned_pool")
            self.assertEqual(routing["dimensions"]["hand_malformation"]["strategy"], "existing_special_pool")
            self.assertEqual(
                routing["dimensions"]["body_proportion_error"]["shared_pool_family"],
                "human_full_body_realistic",
            )
            self.assertEqual(
                routing["dimensions"]["object_structure_error"]["strategy"],
                "rule_recall_then_llm_screen",
            )
            self.assertEqual(
                routing["dimensions"]["material_mismatch"]["shared_pool_family"],
                "structured_object_primary",
            )

            screening_plan = json.loads(expected_files["prompt_pool_screening_plan"].read_text(encoding="utf-8"))
            self.assertIn("object_structure_error", screening_plan["dimensions"])
            self.assertTrue(screening_plan["dimensions"]["object_structure_error"]["requires_llm_screen"])
            self.assertEqual(
                screening_plan["dimensions"]["scale_inconsistency"]["rule_recall_tags"],
                ["has_multiple_objects"],
            )
            self.assertEqual(
                screening_plan["dimensions"]["body_proportion_error"]["shared_pool_family"],
                "human_full_body_realistic",
            )
            self.assertEqual(
                screening_plan["dimensions"]["body_proportion_error"]["turbo_pool_strategy"],
                "reuse_common_semantic_screen_then_token_filter",
            )
            self.assertEqual(
                screening_plan["dimensions"]["penetration_overlap"]["shared_pool_family"],
                "multi_object_reference",
            )
            self.assertIn("human_full_body_realistic", screening_plan["shared_pool_families"])
            self.assertIn("structured_object_primary", screening_plan["shared_pool_families"])
            self.assertIn("multi_object_reference", screening_plan["shared_pool_families"])

            screening_spec = json.loads(expected_files["prompt_pool_screening_spec"].read_text(encoding="utf-8"))
            self.assertIn("shared_pool_families", screening_spec)
            self.assertEqual(
                screening_spec["shared_pool_families"]["human_full_body_realistic"]["builder_mode"],
                "rule_recall_then_llm_screen",
            )
            self.assertFalse(
                screening_spec["dimensions"]["floating_objects"]["requires_llm_screen"]
            )
            self.assertEqual(
                screening_spec["dimensions"]["material_mismatch"]["shared_pool_family"],
                "structured_object_primary",
            )

            build_targets = json.loads(expected_files["prompt_pool_build_targets"].read_text(encoding="utf-8"))
            self.assertEqual(
                build_targets["shared_pool_family_outputs"]["common"]["directory_name"],
                "shared_family_screened_pools_cleaned_v1",
            )
            self.assertEqual(
                build_targets["shared_pool_family_outputs"]["turbo"]["directory_name"],
                "sd35_turbo_shared_family_screened_pools_clipsafe_v1",
            )
            self.assertEqual(
                build_targets["shared_pool_family_outputs"]["turbo"]["derivation_mode"],
                "reuse_common_passes_then_token_filter",
            )
            self.assertEqual(
                build_targets["shared_pool_family_outputs"]["common"]["candidate_suffix"],
                "_candidates.jsonl",
            )
            self.assertEqual(
                build_targets["shared_pool_family_outputs"]["common"]["screening_input_suffix"],
                "_screening_input.jsonl",
            )
            self.assertEqual(
                build_targets["dimension_subpool_outputs"]["common"]["directory_name"],
                "semantic_screened_dimension_subpools_cleaned_v1",
            )
            self.assertEqual(
                build_targets["dimension_subpool_outputs"]["turbo"]["directory_name"],
                "sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1",
            )
            self.assertEqual(
                build_targets["dimensions"]["floating_objects"]["build_mode"],
                "rule_recall_only",
            )

            cleanup = json.loads(expected_files["prompt_pool_cleanup_candidates"].read_text(encoding="utf-8"))
            self.assertIn("delete_requires_confirmation", cleanup)
            self.assertIn("stale_candidate", cleanup)
            self.assertTrue(cleanup["delete_requires_confirmation"])

            top_manifest = json.loads(expected_files["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(top_manifest["artifacts"], {key: str(path) for key, path in expected_files.items()})
            self.assertEqual(result["artifacts"], top_manifest["artifacts"])


if __name__ == "__main__":
    unittest.main()
