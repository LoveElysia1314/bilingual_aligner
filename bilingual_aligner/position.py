"""
Position and line-number utilities for bilingual_aligner.

This module consolidates the responsibilities previously split across
`location_range.py` and `line_number_manager.py` into a single cohesive
module to improve discoverability and reduce file-level fragmentation.

Exports:
 - `LocationRange` dataclass: unified location range representation
 - `LineNumberMapping` dataclass: mapping from filtered indices to original
 - `build_line_number_mapping(...)`: helper to construct mappings from corpus state

This file aims to keep the same public API as the previous modules so that
existing imports continue to work after switching to `from .position import ...`.
"""

from typing import Tuple, Optional, List, Union, Dict
from dataclasses import dataclass


@dataclass
class LocationRange:
    """
    Unified representation of a location range in bilingual aligned text.
    """

    is_source: bool
    line_numbers: Tuple[int, ...]
    content_indices: Optional[Tuple[int, ...]] = None

    def __str__(self) -> str:
        prefix = "src" if self.is_source else "tgt"

        if not self.line_numbers:
            line_part = ""
        elif len(self.line_numbers) == 1:
            line_part = f"{self.line_numbers[0]}"
        else:
            # For compactness and to reduce log size, represent multiple
            # line numbers as a single start-end range (e.g., 193-431).
            # This intentionally collapses non-consecutive lists into a
            # continuous range using the minimum and maximum values.
            line_part = f"{min(self.line_numbers)}-{max(self.line_numbers)}"

        if self.content_indices:
            # Compress content indices similarly for compact display
            if len(self.content_indices) == 1:
                index_part = f"[{self.content_indices[0]}]"
            else:
                # Same compact representation for content indices: use
                # a single start-end range to reduce verbosity.
                index_part = f"[{min(self.content_indices)}-{max(self.content_indices)}]"
            return f"{prefix}@{{{line_part}}}{index_part}"
        else:
            return f"{prefix}@{{{line_part}}}"

    def __repr__(self) -> str:
        return (
            f"LocationRange(is_source={self.is_source}, "
            f"line_numbers={self.line_numbers}, "
            f"content_indices={self.content_indices})"
        )

    def to_dict(self) -> dict:
        return {
            "formatted": str(self),
            "line_numbers": list(self.line_numbers) if self.line_numbers else [],
            "content_indices": (
                list(self.content_indices) if self.content_indices else None
            ),
            "is_source": self.is_source,
        }

    @staticmethod
    def from_original_lines(
        is_source: bool,
        line_numbers: Union[Tuple[int, ...], List[int], int],
    ) -> "LocationRange":
        if isinstance(line_numbers, int):
            line_numbers = (line_numbers,)
        elif isinstance(line_numbers, list):
            line_numbers = tuple(line_numbers)

        return LocationRange(
            is_source=is_source, line_numbers=line_numbers, content_indices=None
        )


@dataclass
class LineNumberMapping:
    """
    Track line number mappings between original files and working state
    """

    source_content_lines: int


def build_line_number_mapping(
    source_lines,
    target_lines,
    source_with_empty: List[str],
    target_with_empty: List[str],
    content_line_map_src: Dict[int, int],
    content_line_map_tgt: Optional[Dict[int, int]] = None,
) -> LineNumberMapping:
    if content_line_map_tgt is None:
        content_line_map_tgt = {}
        orig_line_num = 1
        for filtered_idx, target_line in enumerate(target_lines):
            content_line_map_tgt[filtered_idx] = orig_line_num
            orig_line_num += 1

    return LineNumberMapping(
        source_content_lines=len(source_lines),
    )
