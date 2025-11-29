"""
基于LLM的提示词退化生成器
使用GPT-4o等大语言模型生成质量退化的负样本提示词

优势：
1. 更高的多样性和自然性
2. 可处理复杂的对齐度(alignment)退化
3. 语言流畅，上下文感知

输入层级：
- 子类别(subcategory)级别：low_visual_quality, aesthetic_quality, semantic_plausibility,
  basic_recognition, attribute_alignment, composition_interaction, external_knowledge
- 属性(attribute)级别：blur, noise, color, shape 等具体属性

支持通过 YAML 配置文件加载精细化的 prompt 模板
"""

import os
import json
import time
import logging
import yaml
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re

# OpenAI API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. Install with: pip install openai")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMPromptDegradation:
    """基于LLM的提示词退化生成器"""

    def __init__(
        self,
        llm_config_path: str,
        quality_dimensions_path: str
    ):
        """
        初始化LLM退化生成器

        Args:
            llm_config_path: LLM配置文件路径 (llm_config.yaml)
            quality_dimensions_path: 质量维度配置路径 (quality_dimensions.json)
        """
        self.llm_config_path = llm_config_path
        self.quality_dimensions_path = quality_dimensions_path

        # 加载配置
        self.config = self._load_llm_config()
        self.dimensions = self._load_quality_dimensions()

        # 初始化LLM客户端
        self.client = self._init_llm_client()

        # 退化程度分布（与关键词方法保持一致）
        self.severity_distribution = {
            "mild": 0.2,
            "moderate": 0.4,
            "severe": 0.4
        }

        # 子类别描述映射
        self.subcategory_descriptions = self._build_subcategory_descriptions()

        # 加载 YAML prompt 模板（属性级别）
        self.prompt_templates = self._load_prompt_templates()

        # System Prompt 缓存（性能优化）
        self.system_prompt_cache = {}
        self._build_system_prompt_cache()

        logger.info(f"LLM退化生成器初始化完成 - Model: {self.config['llm']['model']}")
        logger.info(f"System Prompt 缓存: {len(self.system_prompt_cache)} 个配置已预构建")

    def _load_llm_config(self) -> Dict:
        """加载LLM配置"""
        with open(self.llm_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 处理API key：优先使用环境变量，其次使用配置文件
        api_key = config['llm']['api_key']
        
        # 如果配置文件中使用了环境变量占位符
        if api_key and api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"环境变量 {env_var} 未设置")
            config['llm']['api_key'] = api_key
        # 如果配置文件中直接写了key，使用配置文件的
        elif api_key and not api_key.startswith('sk-'):
            # 可能是占位符或无效key
            env_key = os.getenv('OPENAI_API_KEY')
            if env_key:
                logger.info("使用环境变量 OPENAI_API_KEY")
                config['llm']['api_key'] = env_key
            else:
                raise ValueError("API key 无效，请在配置文件中设置或使用环境变量 OPENAI_API_KEY")
        # 如果配置文件中有有效的key，直接使用
        elif api_key and api_key.startswith('sk-'):
            logger.info("使用配置文件中的 API key")
        else:
            # 尝试从环境变量获取
            env_key = os.getenv('OPENAI_API_KEY')
            if env_key:
                logger.info("使用环境变量 OPENAI_API_KEY")
                config['llm']['api_key'] = env_key
            else:
                raise ValueError("未找到 API key，请在配置文件或环境变量中设置")

        return config

    def _load_quality_dimensions(self) -> Dict:
        """加载质量维度配置"""
        with open(self.quality_dimensions_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _init_llm_client(self):
        """初始化LLM客户端"""
        provider = self.config['llm']['provider']

        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("请安装OpenAI库: pip install openai")

            api_key = self.config['llm']['api_key']
            if not api_key:
                raise ValueError("API key 未设置")

            # 获取base_url配置（如果有）
            base_url = self.config['llm'].get('api_base')
            
            logger.info(f"初始化 OpenAI 客户端 - Model: {self.config['llm']['model']}")
            if base_url:
                logger.info(f"使用自定义 API Base: {base_url}")
                return OpenAI(api_key=api_key, base_url=base_url)
            else:
                return OpenAI(api_key=api_key)
        else:
            raise ValueError(f"不支持的LLM provider: {provider}")

    def _build_subcategory_descriptions(self) -> Dict[str, Dict]:
        """
        构建子类别描述信息
        从quality_dimensions.json提取每个子类别的详细信息
        """
        descriptions = {}

        for category, category_data in self.dimensions.items():
            for subcategory, subcategory_data in category_data.get('subcategories', {}).items():
                attributes = subcategory_data.get('attributes', {})

                descriptions[subcategory] = {
                    'category': category,
                    'description': subcategory_data.get('description', ''),
                    'attributes': list(attributes.keys()),
                    'attribute_details': attributes
                }

        return descriptions

    def _load_prompt_templates(self) -> Dict:
        """
        加载 YAML prompt 模板（属性级别精细化模板）

        Returns:
            模板字典，结构为: {subcategory: {attribute: {severity: prompt}}}
        """
        templates = {}
        template_dir = Path(self.llm_config_path).parent / "prompt_templates"

        if not template_dir.exists():
            logger.warning(f"Prompt 模板目录不存在: {template_dir}")
            return templates

        for yaml_file in template_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    file_templates = yaml.safe_load(f)
                    if file_templates:
                        # 合并到总模板中
                        for subcategory, attributes in file_templates.items():
                            if subcategory not in templates:
                                templates[subcategory] = {}
                            templates[subcategory].update(attributes)
                logger.info(f"加载模板文件: {yaml_file.name}")
            except Exception as e:
                logger.error(f"加载模板文件失败 {yaml_file}: {e}")

        # 统计模板数量
        total_prompts = sum(
            len(severities)
            for attrs in templates.values()
            for severities in attrs.values()
        )
        logger.info(f"共加载 {len(templates)} 个子类别, {total_prompts} 个属性级别 prompts")

        return templates

    # 通用原则（所有属性共享，拼接在 specific_instructions 后面）
    COMMON_PRINCIPLES = """
# Degradation Principles
1. Modifications must be based on replacing or adding adjectives/modifiers to achieve degradation
2. Each generation should randomly select or combine different modification methods within this dimension, not using fixed examples
3. Other unspecified quality dimensions should remain unaffected
4. Ensure the generated prompt is linguistically natural and structurally complete

# Output Principles
1. Return only the modified prompt
2. Ensure degradation occurs in the user-specified quality dimension
3. Each call must select different modification methods within this quality dimension to avoid repetitive outputs
"""

    def _build_system_prompt_cache(self):
        """
        预构建并缓存所有 System Prompts（性能优化）

        支持两个层级的缓存：
        1. 属性级别: {subcategory}_{attribute}_{severity} - 使用 YAML 模板
        2. 子类别级别: {subcategory}_{severity} - 使用动态构建（回退）

        缓存数量:
        - 属性级别: ~99个 (33属性 × 3退化程度)
        - 子类别级别: 21个 (7子类别 × 3退化程度)
        """
        logger.info("正在预构建 System Prompt 缓存...")

        # 1. 从 YAML 模板加载属性级别的 prompts（拼接通用原则）
        attribute_count = 0
        for subcategory, attributes in self.prompt_templates.items():
            for attribute, severities in attributes.items():
                for severity, prompt in severities.items():
                    cache_key = f"{subcategory}_{attribute}_{severity}"
                    # YAML 模板 + 通用原则
                    self.system_prompt_cache[cache_key] = prompt + self.COMMON_PRINCIPLES
                    attribute_count += 1

        logger.info(f"属性级别缓存: {attribute_count} 个 prompts")

        # 2. 构建子类别级别的 prompts（回退用）
        subcategory_count = 0
        for subcategory in self.subcategory_descriptions.keys():
            for severity in ['mild', 'moderate', 'severe']:
                cache_key = f"{subcategory}_{severity}"
                # 只有当不存在时才构建（避免覆盖）
                if cache_key not in self.system_prompt_cache:
                    self.system_prompt_cache[cache_key] = self._build_system_prompt(
                        subcategory, severity
                    )
                    subcategory_count += 1

        logger.info(f"子类别级别缓存: {subcategory_count} 个 prompts（回退用）")
        logger.info(f"System Prompt 缓存构建完成: 共 {len(self.system_prompt_cache)} 个配置")

    def _build_system_prompt(self, subcategory: str, severity: str) -> str:
        """
        构建简单的回退 System Prompt（当 YAML 模板不存在时）

        Args:
            subcategory: 子类别名称（如 low_visual_quality）
            severity: 退化程度（mild, moderate, severe）

        Returns:
            System prompt字符串
        """
        subcategory_info = self.subcategory_descriptions.get(subcategory, {})
        category = subcategory_info.get('category', '')

        # Severity descriptions
        severity_desc = {
            "mild": "subtle, barely noticeable",
            "moderate": "clearly noticeable",
            "severe": "dramatic, immediately apparent"
        }.get(severity, "moderate")

        # 简单的回退 System Prompt
        system_prompt = f"""# Task
You are given a text-to-image prompt. Your task is to create degradation in the {subcategory} dimension at {severity} level ({severity_desc}).

Modify the prompt to introduce {severity} level issues in the {subcategory} dimension while keeping the scene content recognizable.

{self.COMMON_PRINCIPLES}"""

        return system_prompt





    def generate_negative_prompt(
        self,
        positive_prompt: str,
        subcategory: str,
        attribute: Optional[str] = None,
        severity: str = "moderate"
    ) -> Tuple[str, Dict]:
        """
        使用LLM生成退化的负样本提示词

        Args:
            positive_prompt: 正样本提示词
            subcategory: 子类别名称（如 low_visual_quality）
            attribute: 具体属性（如 blur, noise）- 可选，不指定则随机选择
            severity: 退化程度 (mild, moderate, severe)

        Returns:
            (负样本提示词, 退化信息字典)
        """
        # 如果未指定属性，随机选择该子类别下的一个属性
        selected_attribute = attribute
        if selected_attribute is None:
            # 尝试从 YAML 模板获取可用属性
            if subcategory in self.prompt_templates:
                available_attrs = list(self.prompt_templates[subcategory].keys())
                if available_attrs:
                    selected_attribute = random.choice(available_attrs)

            # 如果没有 YAML 模板，从 quality_dimensions 获取
            if selected_attribute is None:
                subcategory_info = self.subcategory_descriptions.get(subcategory, {})
                available_attrs = subcategory_info.get('attributes', [])
                if available_attrs:
                    selected_attribute = random.choice(available_attrs)

        # 尝试获取属性级别的 System Prompt
        system_prompt = None
        if selected_attribute:
            attr_cache_key = f"{subcategory}_{selected_attribute}_{severity}"
            system_prompt = self.system_prompt_cache.get(attr_cache_key)

        # 回退到子类别级别
        if system_prompt is None:
            cache_key = f"{subcategory}_{severity}"
            system_prompt = self.system_prompt_cache.get(
                cache_key,
                self._build_system_prompt(subcategory, severity)
            )

        # Build User Prompt (精简版，核心信息由 System Prompt 提供)
        dimension_desc = f"{subcategory}"
        if selected_attribute:
            dimension_desc = f"{subcategory} -> {selected_attribute}"

        user_prompt = f"""**Input**: {positive_prompt}
**Dimension**: {dimension_desc}
**Severity**: {severity}"""

        # 调用LLM API
        try:
            negative_prompt = self._call_llm_api(system_prompt, user_prompt)

            # 验证输出
            if self.config['degradation'].get('validate_output', True):
                negative_prompt = self._validate_and_fix_output(
                    negative_prompt, positive_prompt
                )

            # 退化信息
            subcategory_info = self.subcategory_descriptions.get(subcategory, {})
            degradation_info = {
                "category": subcategory_info.get('category', ''),
                "subcategory": subcategory,
                "attribute": selected_attribute,
                "severity": severity,
                "method": "llm"
            }

            return negative_prompt, degradation_info

        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            raise

    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用LLM API

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词

        Returns:
            LLM生成的负样本提示词
        """
        provider = self.config['llm']['provider']
        max_retries = self.config['llm'].get('max_retries', 3)
        retry_delay = self.config['llm'].get('retry_delay', 2)

        for attempt in range(max_retries):
            try:
                if provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.config['llm']['model'],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.config['llm'].get('temperature', 0.7),
                        max_tokens=self.config['llm'].get('max_tokens', 150),
                        top_p=self.config['llm'].get('top_p', 0.95)
                    )

                    negative_prompt = response.choices[0].message.content.strip()

                    # 记录API调用
                    if self.config['logging'].get('log_api_calls', False):
                        logger.debug(f"API调用成功 - Tokens: {response.usage.total_tokens}")

                    return negative_prompt

                else:
                    raise ValueError(f"不支持的provider: {provider}")

            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def _validate_and_fix_output(
        self,
        negative_prompt: str,
        positive_prompt: str
    ) -> str:
        """
        验证并修复LLM输出

        Args:
            negative_prompt: LLM生成的负样本
            positive_prompt: 原正样本

        Returns:
            验证并修复后的负样本
        """
        # 移除可能的前缀
        negative_prompt = re.sub(
            r'^(negative prompt|输出|结果)[:：]\s*',
            '',
            negative_prompt,
            flags=re.IGNORECASE
        )

        # 移除引号
        negative_prompt = negative_prompt.strip('"\'')

        # 检查长度
        min_length = self.config['degradation'].get('min_length', 10)
        max_length = self.config['degradation'].get('max_length', 200)
        word_count = len(negative_prompt.split())

        if word_count < min_length or word_count > max_length:
            logger.warning(f"负样本长度异常: {word_count} 词")

        # 确保与正样本有差异
        if self.config['degradation'].get('ensure_modification', True):
            if negative_prompt.lower() == positive_prompt.lower():
                logger.warning("负样本与正样本完全相同，可能生成失败")

        return negative_prompt.strip()

    def get_all_subcategories(self) -> List[Dict]:
        """
        获取所有子类别（用于数据集生成）

        Returns:
            子类别信息列表，每个元素包含 subcategory 和 category
        """
        subcategories = []
        for subcategory, info in self.subcategory_descriptions.items():
            subcategories.append({
                'subcategory': subcategory,
                'category': info['category'],
                'description': info['description']
            })
        return subcategories

    def select_severity_random(self) -> str:
        """
        根据分布随机选择退化程度

        Returns:
            severity: mild, moderate, severe
        """
        import random
        return random.choices(
            list(self.severity_distribution.keys()),
            weights=list(self.severity_distribution.values()),
            k=1
        )[0]

    def generate_batch_negatives(
        self,
        positive_prompts: List[str],
        subcategories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None
    ) -> List[Tuple[str, str, Dict]]:
        """
        批量生成负样本

        Args:
            positive_prompts: 正样本列表
            subcategories: 子类别列表（如果为None，随机选择）
            severities: 退化程度列表（如果为None，根据分布随机选择）

        Returns:
            (正样本, 负样本, 退化信息) 的列表
        """
        import random

        results = []
        all_subcategories = list(self.subcategory_descriptions.keys())

        for i, positive_prompt in enumerate(positive_prompts):
            # 选择子类别
            if subcategories and i < len(subcategories):
                subcategory = subcategories[i]
            else:
                subcategory = random.choice(all_subcategories)

            # 选择退化程度
            if severities and i < len(severities):
                severity = severities[i]
            else:
                severity = random.choices(
                    list(self.severity_distribution.keys()),
                    weights=list(self.severity_distribution.values()),
                    k=1
                )[0]

            try:
                negative_prompt, degradation_info = self.generate_negative_prompt(
                    positive_prompt, subcategory, severity
                )
                results.append((positive_prompt, negative_prompt, degradation_info))
            except Exception as e:
                logger.error(f"批量生成第{i}个样本失败: {e}")
                continue

        return results


if __name__ == "__main__":
    # 测试代码
    import sys

    # 配置路径
    llm_config_path = "/root/ImageReward/data_generation/config/llm_config.yaml"
    quality_dimensions_path = "/root/ImageReward/data_generation/config/quality_dimensions.json"

    # 检查API key
    if not os.getenv("OPENAI_API_KEY"):
        print("错误: 请设置OPENAI_API_KEY环境变量")
        print("使用方法: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # 创建生成器（不使用fallback）
    print("初始化LLM退化生成器...")
    generator = LLMPromptDegradation(
        llm_config_path=llm_config_path,
        quality_dimensions_path=quality_dimensions_path
    )

    # 测试不同子类别
    test_cases = [
        {
            "prompt": "a beautiful sunset over the ocean, masterpiece, high quality, detailed",
            "subcategory": "low_visual_quality",
            "severity": "moderate"
        },
        {
            "prompt": "a red apple on a wooden table, professional photography",
            "subcategory": "attribute_alignment",
            "severity": "moderate"
        },
        {
            "prompt": "portrait of a smiling woman, natural lighting, sharp focus",
            "subcategory": "semantic_plausibility",
            "severity": "mild"
        }
    ]

    print("\n" + "=" * 80)
    print("测试LLM退化生成")
    print("=" * 80)

    for i, test in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】")
        print(f"子类别: {test['subcategory']}")
        print(f"退化程度: {test['severity']}")
        print(f"正样本: {test['prompt']}")

        try:
            negative_prompt, degradation_info = generator.generate_negative_prompt(
                test['prompt'],
                test['subcategory'],
                test['severity']
            )

            print(f"负样本: {negative_prompt}")
            print(f"退化信息: {degradation_info}")
        except Exception as e:
            print(f"生成失败: {e}")

    print("\n" + "=" * 80)
