"""对比分析器

从 tools/compare_models.py 迁移。
对比不同的Analysis results。
"""

from typing import Dict, Any, List

from .base import ComparisonAnalyzer, AnalysisResult
from .similarity import TextSimilarityAnalyzer
from .encoding import TextEncodingAnalyzer
from .punctuation import TextPunctuationAnalyzer


class MultiMethodComparisonAnalyzer(ComparisonAnalyzer):
    """多方法对比分析器

    使用多个分析器对比Analysis results。
    """

    def __init__(self, processor=None):
        super().__init__(processor)
        self.similarity_analyzer = TextSimilarityAnalyzer(processor)
        self.encoding_analyzer = TextEncodingAnalyzer(processor)
        self.punctuation_analyzer = TextPunctuationAnalyzer(processor)

    def analyze(self, text1: str, text2: str) -> AnalysisResult:
        """执行多方法对比分析

        Args:
            text1: 第一段文本
            text2: 第二段文本

        Returns:
            Analysis results
        """
        data = {}

        # 执行相似度分析
        similarity_result = self.similarity_analyzer.analyze_detailed(text1, text2)
        data["similarity"] = similarity_result.data

        # 执行编码方法分析
        encoding_result = self.encoding_analyzer.analyze_detailed(text1, text2)
        data["encoding"] = encoding_result.data

        # 执行编码方法对比
        method_comparison = self.encoding_analyzer.compare_methods(text1, text2)
        data["encoding_methods"] = method_comparison.data

        # 执行标点符号分析
        punctuation_result = self.punctuation_analyzer.analyze(text1, text2)
        data["punctuation"] = punctuation_result.data

        return AnalysisResult(
            tool_name="comparison",
            result_type="comprehensive",
            data=data,
        )

    def analyze_quick(self, text1: str, text2: str) -> AnalysisResult:
        """快速对比分析"""
        data = {}

        # 只做必要的分析
        similarity_result = self.similarity_analyzer.analyze_quick(text1, text2)
        data["similarity"] = similarity_result.data

        encoding_result = self.encoding_analyzer.analyze_quick(text1, text2)
        data["encoding"] = encoding_result.data

        return AnalysisResult(
            tool_name="comparison",
            result_type="quick",
            data=data,
        )

    def analyze_detailed(self, text1: str, text2: str) -> AnalysisResult:
        """详细对比分析"""
        return self.analyze(text1, text2)
