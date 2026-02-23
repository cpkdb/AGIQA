#!/usr/bin/env python3
"""
KnowledgeBase (Stage 6)
持久化的跨运行知识库，记录维度统计、策略效果、模型兼容性。

支持:
- 每次 VLM 判定后更新统计
- 维度健康度报告
- Circuit Breaker 防卡死
- 断点续跑（JSON 持久化）

可独立运行查看报告:
    python scripts/knowledge_base.py --report outputs/knowledge_base/
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBase:
    """持久化的跨运行知识库"""

    DEFAULT_EVOLUTION_THRESHOLD = 0.6
    DEFAULT_INCOMPATIBLE_THRESHOLD = 0.3
    DEFAULT_MIN_ATTEMPTS_EVOLUTION = 5
    DEFAULT_MIN_ATTEMPTS_COMPAT = 10
    DEFAULT_CIRCUIT_BREAKER_LIMIT = 10

    def __init__(
        self,
        path: str = "outputs/knowledge_base/",
        evolution_threshold: float = DEFAULT_EVOLUTION_THRESHOLD,
        incompatible_threshold: float = DEFAULT_INCOMPATIBLE_THRESHOLD,
        min_attempts_evolution: int = DEFAULT_MIN_ATTEMPTS_EVOLUTION,
        min_attempts_compat: int = DEFAULT_MIN_ATTEMPTS_COMPAT,
        circuit_breaker_limit: int = DEFAULT_CIRCUIT_BREAKER_LIMIT,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.evolution_threshold = evolution_threshold
        self.incompatible_threshold = incompatible_threshold
        self.min_attempts_evolution = min_attempts_evolution
        self.min_attempts_compat = min_attempts_compat
        self.circuit_breaker_limit = circuit_breaker_limit

        # {dim: {attempts, successes, total_scores, consecutive_failures}}
        self.dimension_stats: Dict[str, Dict] = {}
        # {"dim|sev|model": {template_id, attempts, successes}}
        self.strategy_stats: Dict[str, Dict] = {}
        # {"model|dim": {attempts, successes}}
        self.compat_matrix: Dict[str, Dict] = {}
        # 暂停的维度（Circuit Breaker 触发）
        self.paused_dimensions: Set[str] = set()

        self.load()

    # ------------------------------------------------------------------ #
    #  Core API
    # ------------------------------------------------------------------ #

    def report_outcome(
        self,
        dimension: str,
        severity: str,
        model_id: str,
        template_id: Optional[str],
        success: bool,
        scores: Optional[Dict[str, float]] = None,
        failure_type: Optional[str] = None,
    ):
        """每次 VLM 判定后更新知识库"""
        # 1. dimension_stats
        ds = self.dimension_stats.setdefault(
            dimension,
            {"attempts": 0, "successes": 0, "total_scores": {}, "consecutive_failures": 0, "failure_types": {}},
        )
        ds["attempts"] += 1
        if success:
            ds["successes"] += 1
            ds["consecutive_failures"] = 0
        else:
            ds["consecutive_failures"] = ds.get("consecutive_failures", 0) + 1
            if failure_type:
                ds["failure_types"][failure_type] = ds["failure_types"].get(failure_type, 0) + 1

        # 累加多维度评分
        if scores:
            ts = ds.setdefault("total_scores", {})
            for k, v in scores.items():
                ts[k] = ts.get(k, 0.0) + v

        # Circuit Breaker
        if ds["consecutive_failures"] >= self.circuit_breaker_limit:
            if dimension not in self.paused_dimensions:
                self.paused_dimensions.add(dimension)
                logger.warning(
                    f"[CircuitBreaker] 维度 '{dimension}' 连续失败 {ds['consecutive_failures']} 次，已暂停"
                )

        # 2. strategy_stats
        if template_id:
            skey = f"{dimension}|{severity}|{model_id}"
            ss = self.strategy_stats.setdefault(
                skey, {"template_id": template_id, "attempts": 0, "successes": 0}
            )
            ss["template_id"] = template_id
            ss["attempts"] += 1
            if success:
                ss["successes"] += 1

        # 3. compat_matrix
        ckey = f"{model_id}|{dimension}"
        cm = self.compat_matrix.setdefault(ckey, {"attempts": 0, "successes": 0})
        cm["attempts"] += 1
        if success:
            cm["successes"] += 1

    def is_dimension_paused(self, dimension: str) -> bool:
        """检查维度是否被 Circuit Breaker 暂停"""
        return dimension in self.paused_dimensions

    def reset_dimension_pause(self, dimension: str):
        """手动恢复暂停的维度"""
        self.paused_dimensions.discard(dimension)
        ds = self.dimension_stats.get(dimension)
        if ds:
            ds["consecutive_failures"] = 0

    def is_model_compatible(self, model_id: str, dimension: str) -> bool:
        """检查模型-维度兼容性"""
        ckey = f"{model_id}|{dimension}"
        cm = self.compat_matrix.get(ckey)
        if cm is None:
            return True  # 无数据，默认兼容
        if cm["attempts"] < self.min_attempts_compat:
            return True  # 数据不足，不做判断
        rate = cm["successes"] / cm["attempts"]
        return rate >= self.incompatible_threshold

    def should_trigger_evolution(self, dimension: str, severity: str, model_id: str) -> bool:
        """是否应触发策略进化"""
        skey = f"{dimension}|{severity}|{model_id}"
        ss = self.strategy_stats.get(skey)
        if ss is None or ss["attempts"] < self.min_attempts_evolution:
            return False
        rate = ss["successes"] / ss["attempts"]
        return rate < self.evolution_threshold

    def get_dimension_health(self) -> Dict:
        """返回各维度的健康度报告"""
        report = {}
        for dim, ds in self.dimension_stats.items():
            attempts = ds["attempts"]
            successes = ds["successes"]
            rate = successes / attempts if attempts > 0 else 0.0

            avg_scores = {}
            ts = ds.get("total_scores", {})
            for k, v in ts.items():
                avg_scores[k] = round(v / attempts, 3) if attempts > 0 else 0.0

            report[dim] = {
                "attempts": attempts,
                "successes": successes,
                "success_rate": round(rate, 4),
                "consecutive_failures": ds.get("consecutive_failures", 0),
                "paused": dim in self.paused_dimensions,
                "avg_scores": avg_scores,
                "failure_types": ds.get("failure_types", {}),
            }
        return report

    def get_compat_report(self) -> Dict:
        """返回模型-维度兼容性报告"""
        report = {}
        for ckey, cm in self.compat_matrix.items():
            attempts = cm["attempts"]
            rate = cm["successes"] / attempts if attempts > 0 else 0.0
            report[ckey] = {
                "attempts": attempts,
                "successes": cm["successes"],
                "success_rate": round(rate, 4),
                "compatible": rate >= self.incompatible_threshold or attempts < self.min_attempts_compat,
            }
        return report

    def get_strategy_report(self) -> Dict:
        """返回策略效果报告"""
        report = {}
        for skey, ss in self.strategy_stats.items():
            attempts = ss["attempts"]
            rate = ss["successes"] / attempts if attempts > 0 else 0.0
            report[skey] = {
                "template_id": ss.get("template_id"),
                "attempts": attempts,
                "successes": ss["successes"],
                "success_rate": round(rate, 4),
                "needs_evolution": rate < self.evolution_threshold and attempts >= self.min_attempts_evolution,
            }
        return report

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self):
        """持久化到磁盘"""
        data = {
            "saved_at": datetime.now().isoformat(),
            "dimension_stats": self.dimension_stats,
            "strategy_stats": self.strategy_stats,
            "compat_matrix": self.compat_matrix,
            "paused_dimensions": list(self.paused_dimensions),
        }
        with open(self.path / "knowledge_base.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """从磁盘恢复"""
        kb_path = self.path / "knowledge_base.json"
        if not kb_path.exists():
            return

        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.dimension_stats = data.get("dimension_stats", {})
            self.strategy_stats = data.get("strategy_stats", {})
            self.compat_matrix = data.get("compat_matrix", {})
            self.paused_dimensions = set(data.get("paused_dimensions", []))
            logger.info(
                f"KnowledgeBase loaded: {len(self.dimension_stats)} dims, "
                f"{len(self.compat_matrix)} compat entries, "
                f"{len(self.paused_dimensions)} paused"
            )
        except Exception as e:
            logger.warning(f"Failed to load KnowledgeBase: {e}")

    # ------------------------------------------------------------------ #
    #  Report (CLI)
    # ------------------------------------------------------------------ #

    def print_report(self):
        """打印完整报告"""
        health = self.get_dimension_health()
        compat = self.get_compat_report()
        strategy = self.get_strategy_report()

        print("\n" + "=" * 60)
        print("  KnowledgeBase Report")
        print("=" * 60)

        # 维度健康度
        print(f"\n{'维度':<30} {'尝试':>6} {'成功':>6} {'成功率':>8} {'状态':>8}")
        print("-" * 62)
        for dim in sorted(health.keys()):
            h = health[dim]
            status = "暂停" if h["paused"] else "正常"
            print(
                f"{dim:<30} {h['attempts']:>6} {h['successes']:>6} "
                f"{h['success_rate']*100:>7.1f}% {status:>8}"
            )

        # 兼容性矩阵
        if compat:
            print(f"\n{'模型|维度':<40} {'尝试':>6} {'成功率':>8} {'兼容':>6}")
            print("-" * 62)
            for ckey in sorted(compat.keys()):
                c = compat[ckey]
                compat_str = "是" if c["compatible"] else "否"
                print(
                    f"{ckey:<40} {c['attempts']:>6} "
                    f"{c['success_rate']*100:>7.1f}% {compat_str:>6}"
                )

        # 需要进化的策略
        needs_evo = {k: v for k, v in strategy.items() if v.get("needs_evolution")}
        if needs_evo:
            print(f"\n需要进化的策略 ({len(needs_evo)}):")
            for skey, s in sorted(needs_evo.items()):
                print(f"  {skey}: {s['success_rate']*100:.1f}% ({s['attempts']} attempts)")

        if self.paused_dimensions:
            print(f"\n暂停的维度: {', '.join(sorted(self.paused_dimensions))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KnowledgeBase (Stage 6) 报告工具")
    parser.add_argument(
        "--report",
        type=str,
        default="outputs/knowledge_base/",
        help="KnowledgeBase 目录路径",
    )
    args = parser.parse_args()

    kb = KnowledgeBase(path=args.report)
    kb.print_report()
