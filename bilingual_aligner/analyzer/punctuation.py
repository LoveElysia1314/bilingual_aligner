"""标点符号分析器

从 tools/punctuation_analyzer.py 迁移。
分析文本的标点符号特性。
"""

from typing import Dict, Any, Optional, List
from collections import Counter

from .base import PunctuationAnalyzer, AnalysisResult


class TextPunctuationAnalyzer(PunctuationAnalyzer):
    """文本标点符号分析器

    分析文本中的标点符号类型和分布。
    """

    # 中文标点
    CHINESE_PUNCTUATION = set("，。！？；：" "''（）【】《》、……·~")

    # 英文标点
    ENGLISH_PUNCTUATION = set(",.!?;:\"'()[]{}~-")

    def __init__(self, processor=None):
        super().__init__(processor)

    def analyze_punctuation(self, text: str) -> Dict[str, Any]:
        """分析文本的标点符号"""
        chinese_punct = []
        english_punct = []
        other_punct = []

        for char in text:
            if char in self.CHINESE_PUNCTUATION:
                chinese_punct.append(char)
            elif char in self.ENGLISH_PUNCTUATION:
                english_punct.append(char)
            elif not char.isalnum() and not char.isspace():
                other_punct.append(char)

        data = {
            "total_punctuation": len(chinese_punct)
            + len(english_punct)
            + len(other_punct),
            "chinese_punctuation_count": len(chinese_punct),
            "english_punctuation_count": len(english_punct),
            "other_punctuation_count": len(other_punct),
        }

        # 统计出现频率最高的标点
        all_punct = chinese_punct + english_punct + other_punct
        if all_punct:
            counter = Counter(all_punct)
            data["most_common_punctuation"] = dict(counter.most_common(5))

        # 计算标点密度
        text_length = len(text)
        if text_length > 0:
            data["punctuation_density"] = data["total_punctuation"] / text_length
        else:
            data["punctuation_density"] = 0.0

        return data

    def analyze(self, text: str, text2: Optional[str] = None) -> AnalysisResult:
        """分析标点符号

        Args:
            text: 第一段文本
            text2: 第二段文本（可选）

        Returns:
            Analysis results
        """
        data = {}

        # 分析第一段文本
        data["text1_punctuation"] = self.analyze_punctuation(text)

        # 如果提供了第二段文本
        if text2:
            data["text2_punctuation"] = self.analyze_punctuation(text2)

            # 比较两段文本的标点特性
            punct1 = data["text1_punctuation"]
            punct2 = data["text2_punctuation"]

            data["comparison"] = {
                "density_difference": abs(
                    punct1.get("punctuation_density", 0)
                    - punct2.get("punctuation_density", 0)
                ),
                "punctuation_ratio": (
                    punct1.get("total_punctuation", 0)
                    / punct2.get("total_punctuation", 1)
                    if punct2.get("total_punctuation", 0) > 0
                    else 0
                ),
            }

        return AnalysisResult(
            tool_name="punctuation",
            result_type="analysis",
            data=data,
        )

    def calculate_punctuation_weight(self, text1: str, text2: str) -> float:
        """计算标点权重（用于相似度调整）

        Args:
            text1: 第一段文本
            text2: 第二段文本

        Returns:
            权重值
        """
        try:
            from bilingual_aligner.core.punctuation import (
                calculate_punctuation_similarity,
            )

            return calculate_punctuation_similarity(text1, text2)
        except Exception:
            return 1.0
