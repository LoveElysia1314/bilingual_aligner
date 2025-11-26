"""
Core modules for text alignment.
"""

from .processor import TextProcessor, get_text_processor
from .repairer import TextAligner
from .splitter import UniversalSplitter
from .punctuation import (
    PunctuationHandler,
    calculate_punctuation_similarity,
)

__all__ = [
    "TextProcessor",
    "get_text_processor",
    "TextAligner",
    "UniversalSplitter",
    "PunctuationHandler",
    "calculate_punctuation_similarity",
]
