"""
快速测试Demo - 生成少量样本验证流程

使用方法:
    python quick_demo.py
    python quick_demo.py --num_samples 5
    python quick_demo.py --no_reward  # 不使用ImageReward评分
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from contrastive_dataset_demo import ContrastiveDatasetDemo, get_demo_prompts
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_test(num_samples: int = 2, use_reward: bool = True):
    """
    快速测试函数

    Args:
        num_samples: 生成样本数量
        use_reward: 是否使用ImageReward评分
    """
    logger.info("=" * 60)
    logger.info("正负对比自监督AIGC数据集生成 - 快速测试")
    logger.info("=" * 60)

    # 获取少量测试prompts
    prompts = get_demo_prompts()[:num_samples]

    logger.info(f"将生成 {len(prompts)} 个样本进行测试")
    for i, p in enumerate(prompts):
        logger.info(f"  [{i+1}] {p['prompt'][:50]}...")

    # 创建Demo实例
    demo = ContrastiveDatasetDemo(
        output_dir="/root/ImageReward/data_generation/demo_output",
        use_image_reward=use_reward
    )

    try:
        # 运行
        demo.run_demo(prompts=prompts, base_seed=42)

        # 显示结果
        logger.info("\n" + "=" * 60)
        logger.info("测试完成！生成的样本对:")
        logger.info("=" * 60)

        for pair in demo.dataset["pairs"]:
            logger.info(f"\n样本ID: {pair['pair_id']}")
            logger.info(f"  类型: {pair['type']}")
            logger.info(f"  正样本分数: {pair['positive'].get('reward_score', 'N/A')}")
            logger.info(f"  负样本分数: {pair['negative'].get('reward_score', 'N/A')}")
            logger.info(f"  分数差: {pair.get('score_difference', 'N/A')}")

    finally:
        demo.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="快速测试Demo")
    parser.add_argument("--num_samples", type=int, default=2, help="生成样本数量")
    parser.add_argument("--no_reward", action="store_true", help="禁用ImageReward")

    args = parser.parse_args()

    quick_test(num_samples=args.num_samples, use_reward=not args.no_reward)
