"""Analyzer base classes and interfaces

All analysis tools inherit from BaseAnalyzer, implementing a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging


class AnalysisResult:
    """Analysis result container"""

    def __init__(
        self,
        tool_name: str,
        result_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize analysis result

        Args:
            tool_name: Tool name
            result_type: Result type ("similarity", "encoding", "punctuation", etc.)
            data: Analysis data
            metadata: Additional metadata
        """
        self.tool_name = tool_name
        self.result_type = result_type
        self.data = data
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tool": self.tool_name,
            "type": self.result_type,
            "data": self.data,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"AnalysisResult({self.tool_name}, {self.result_type})"


class BaseAnalyzer(ABC):
    """Analyzer base class

    All specific analyzers should inherit from this class and implement abstract methods.

    Responsibilities:
    - Analyze text characteristics
    - Return unified analysis results
    - Support quick mode and detailed mode
    """

    def __init__(self, name: str, processor=None):
        """初始化分析器

        Args:
            name: 分析器名称
            processor: 文本处理器
        """
        self.name = name
        self.processor = processor
        self.logger = logging.getLogger(f"analyzer.{name}")

    @abstractmethod
    def analyze(self, text1: str, text2: Optional[str] = None) -> AnalysisResult:
        """执行分析

        Args:
            text1: 第一段文本
            text2: 第二段文本（可选）

        Returns:
            分析结果
        """
        pass

    def analyze_quick(self, text1: str, text2: Optional[str] = None) -> AnalysisResult:
        """快速分析（默认实现）

        Args:
            text1: 第一段文本
            text2: 第二段文本（可选）

        Returns:
            分析结果
        """
        return self.analyze(text1, text2)

    def analyze_detailed(
        self, text1: str, text2: Optional[str] = None
    ) -> AnalysisResult:
        """详细分析（默认实现）

        Args:
            text1: 第一段文本
            text2: 第二段文本（可选）

        Returns:
            分析结果
        """
        return self.analyze(text1, text2)


class SimilarityAnalyzer(BaseAnalyzer):
    """相似度分析器"""

    def __init__(self, processor=None):
        super().__init__("similarity", processor)

    @abstractmethod
    def analyze(self, text1: str, text2: str) -> AnalysisResult:
        """分析相似度"""
        pass


class EncodingAnalyzer(BaseAnalyzer):
    """编码方法分析器"""

    def __init__(self, processor=None):
        super().__init__("encoding", processor)

    @abstractmethod
    def analyze(self, text1: str, text2: Optional[str] = None) -> AnalysisResult:
        """分析编码方法"""
        pass


class PunctuationAnalyzer(BaseAnalyzer):
    """标点符号分析器"""

    def __init__(self, processor=None):
        super().__init__("punctuation", processor)

    @abstractmethod
    def analyze(self, text: str, text2: Optional[str] = None) -> AnalysisResult:
        """分析标点符号"""
        pass


class ComparisonAnalyzer(BaseAnalyzer):
    """比较分析器"""

    def __init__(self, processor=None):
        super().__init__("comparison", processor)

    @abstractmethod
    def analyze(self, text1: str, text2: str) -> AnalysisResult:
        """执行对比分析"""
        pass
