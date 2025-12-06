"""Analyzer module

Unified text analysis tool framework.

Provides the following analyzers:
- TextSimilarityAnalyzer: Text similarity analysis
- TextEncodingAnalyzer: Encoding method analysis
- TextPunctuationAnalyzer: Punctuation analysis
- MultiMethodComparisonAnalyzer: Multi-method comparison analysis
"""

from .base import (
    BaseAnalyzer,
    SimilarityAnalyzer,
    EncodingAnalyzer,
    PunctuationAnalyzer,
    ComparisonAnalyzer,
    AnalysisResult,
)
from .similarity import TextSimilarityAnalyzer
from .encoding import TextEncodingAnalyzer
from .punctuation import TextPunctuationAnalyzer
from .comparison import MultiMethodComparisonAnalyzer

__all__ = [
    "BaseAnalyzer",
    "SimilarityAnalyzer",
    "EncodingAnalyzer",
    "PunctuationAnalyzer",
    "ComparisonAnalyzer",
    "AnalysisResult",
    "TextSimilarityAnalyzer",
    "TextEncodingAnalyzer",
    "TextPunctuationAnalyzer",
    "MultiMethodComparisonAnalyzer",
]
