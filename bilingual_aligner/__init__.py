"""
Bilingual Aligner: A library for bilingual text alignment and repair.
"""

import logging

from .api import TextAligner, calculate_similarity
from .core.processor import get_text_processor, TextProcessor
from .corpus import RepairType, AlignmentState
from .position import LocationRange
from . import core
from .aligners import EnumPruningDPAligner, TwoStageDPAligner

__version__ = "0.5.0"
__all__ = [
    "TextAligner",
    "calculate_similarity",
    "get_text_processor",
    "TextProcessor",
    "RepairType",
    "AlignmentState",
    "LocationRange",
    "EnumPruningDPAligner",
    "TwoStageDPAligner",
    "core",
]

# Configure default logging format to be minimal
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger = logging.getLogger("bilingual_aligner")
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
