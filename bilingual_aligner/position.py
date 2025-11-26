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
            line_part = ",".join(map(str, self.line_numbers))

        if self.content_indices:
            if len(self.content_indices) == 1:
                index_part = f"[{self.content_indices[0]}]"
            else:
                index_part = f"[{','.join(map(str, self.content_indices))}]"
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

    @staticmethod
    def from_filtered_indices(
        is_source: bool,
        line_numbers: Union[Tuple[int, ...], List[int], int],
        content_indices: Union[Tuple[int, ...], List[int], int],
    ) -> "LocationRange":
        if isinstance(line_numbers, int):
            line_numbers = (line_numbers,)
        elif isinstance(line_numbers, list):
            line_numbers = tuple(line_numbers)

        if isinstance(content_indices, int):
            content_indices = (content_indices,)
        elif isinstance(content_indices, list):
            content_indices = tuple(content_indices)

        return LocationRange(
            is_source=is_source,
            line_numbers=line_numbers,
            content_indices=content_indices,
        )

    @staticmethod
    def from_alignment_step(
        source_start: int,
        source_end: int,
        target_start: int,
        target_end: int,
        source_line_map: Optional[dict] = None,
        target_line_map: Optional[dict] = None,
    ) -> Tuple["LocationRange", "LocationRange"]:
        src_indices = tuple(range(source_start, source_end))
        tgt_indices = tuple(range(target_start, target_end))

        if source_line_map:
            src_lines = tuple(source_line_map.get(i, i + 1) for i in src_indices)
        else:
            src_lines = tuple(i + 1 for i in src_indices)

        if target_line_map:
            tgt_lines = tuple(target_line_map.get(i, i + 1) for i in tgt_indices)
        else:
            tgt_lines = tuple(i + 1 for i in tgt_indices)

        src_range = LocationRange(
            is_source=True, line_numbers=src_lines, content_indices=src_indices
        )
        tgt_range = LocationRange(
            is_source=False, line_numbers=tgt_lines, content_indices=tgt_indices
        )

        return src_range, tgt_range

    @staticmethod
    def from_repair_log(
        source_orig_line_numbers: Optional[Tuple[int, ...]] = None,
        target_orig_line_numbers: Optional[Tuple[int, ...]] = None,
        source_filtered_position: Optional[Tuple[int, int]] = None,
        target_filtered_position: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional["LocationRange"], Optional["LocationRange"]]:
        src_range = None
        tgt_range = None

        if source_orig_line_numbers:
            src_range = LocationRange(
                is_source=True,
                line_numbers=source_orig_line_numbers,
                content_indices=source_filtered_position,
            )

        if target_orig_line_numbers:
            tgt_range = LocationRange(
                is_source=False,
                line_numbers=target_orig_line_numbers,
                content_indices=target_filtered_position,
            )

        return src_range, tgt_range


@dataclass
class LineNumberMapping:
    """
    Track line number mappings between original files and working state
    """

    filtered_to_original_src: Dict[int, int]
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
        filtered_to_original_src=content_line_map_src,
        source_content_lines=len(source_lines),
    )
