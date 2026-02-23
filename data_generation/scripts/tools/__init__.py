"""
smolagents Tool 定义
封装现有的 LLMPromptDegradation 和 SDXLGenerator，新增 degradation_judge
"""

from .prompt_degrader import prompt_degrader
from .image_generator import image_generator
from .degradation_judge import degradation_judge

__all__ = ['prompt_degrader', 'image_generator', 'degradation_judge']
