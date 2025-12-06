"""Base aligner interface

Defines the unified interface for all alignment algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class Alignment:
    """Alignment result data structure

    Represents the result of an alignment operation, including source and target index ranges and alignment quality score.
    """

    source_indices: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self):
        return (
            f"Alignment(src={self.source_indices}, "
            f"tgt={self.target_indices}, score={self.score:.3f})"
        )

    @property
    def is_one_to_one(self) -> bool:
        """Whether it is 1:1 alignment"""
        return len(self.source_indices) == 1 and len(self.target_indices) == 1

    @property
    def operation_type(self) -> str:
        """Alignment operation type"""
        src_len = len(self.source_indices)
        tgt_len = len(self.target_indices)
        if src_len == 1 and tgt_len == 1:
            return "1:1"
        elif src_len == 2 and tgt_len == 1:
            return "2:1"
        elif src_len == 1 and tgt_len == 2:
            return "1:2"
        else:
            return f"{src_len}:{tgt_len}"


class AlignerBase(ABC):
    """Aligner base class

    All alignment algorithms should inherit from this class and implement the align() method.

    Responsibilities:
    - Define unified alignment interface
    - Provide input validation
    - Handle logging
    """

    def __init__(self, **config):
        """Initialize aligner

        Args:
            **config: Algorithm-related configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def align(
        self, source_lines: List[str], target_lines: List[str]
    ) -> List[Alignment]:
        """Perform alignment

        Args:
            source_lines: Source language text line list
            target_lines: Target language text line list

        Returns:
            List of alignment results, each element is an Alignment object

        Raises:
            ValueError: Invalid input parameters
        """
        raise NotImplementedError
