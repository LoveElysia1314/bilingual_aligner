"""
Bilingual Aligner: A library for bilingual text alignment and repair.
"""

import logging

from .api import TextAligner, calculate_similarity
from .core.processor import get_text_processor, TextProcessor
from .repair.models import RepairType, AlignmentState
from .position import LocationRange
from . import core

# Alignment module
from .alignment import (
    AlignerBase,
    Alignment,
    EnumPruningAligner,
)

# Repair module
from . import repair
from .repair import (
    RepairApplier,
    RepairCoordinator,
    RepairStrategy,
)

# Analyzer module
from . import analyzer
from .analyzer import (
    BaseAnalyzer,
    TextSimilarityAnalyzer,
    TextEncodingAnalyzer,
    TextPunctuationAnalyzer,
    MultiMethodComparisonAnalyzer,
)

__version__ = "0.6.0"
__all__ = [
    "TextAligner",
    "calculate_similarity",
    "get_text_processor",
    "TextProcessor",
    "RepairType",
    "AlignmentState",
    "LocationRange",
    "AlignerBase",
    "Alignment",
    "EnumPruningAligner",
    "RepairApplier",
    "RepairCoordinator",
    "RepairStrategy",
    "BaseAnalyzer",
    "TextSimilarityAnalyzer",
    "TextEncodingAnalyzer",
    "TextPunctuationAnalyzer",
    "MultiMethodComparisonAnalyzer",
    "core",
    "repair",
    "analyzer",
]

# Configure default logging format to be minimal
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger = logging.getLogger("bilingual_aligner")
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
