"""
Corpus management for bilingual text alignment.

Note: Data model classes (RepairType, SplitType, AlignmentState, RepairLog)
have been migrated to repair.models. This file now only contains
BilingualCorpus and LineObject classes.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import numpy as np
from .core.processor import TextProcessor
from .core.punctuation import (
    PunctuationHandler,
    calculate_punctuation_similarity,
)


@dataclass
class LineObject:
    """Unified representation of a text line with all metadata and cached features"""

    text: str
    is_source: bool
    hash_value: str = ""
    original_line_number: int = -1  # 1-based, original file position
    current_index: int = (
        -1
    )  # 0-based, position after removing empty lines and after repairs
    embedding: Optional[np.ndarray] = None  # Normalized semantic vector
    sentence_count: Optional[int] = None  # Sentence count after splitting

    def __post_init__(self):
        if not self.hash_value:
            self.hash_value = hashlib.sha256(
                self.text.strip().encode("utf-8")
            ).hexdigest()

    def ensure_features(self, text_processor: TextProcessor):
        """Lazy compute expensive features when needed - using sentence-level encoding"""
        if self.sentence_count is None:
            self.sentence_count = len(text_processor.split_sentences(self.text))
        if self.embedding is None:
            # Use sentence-level encoding for better precision
            self.embedding = text_processor.get_normalized_embedding_by_sentences(
                self.text, method="mean"
            )


class BilingualCorpus:
    """Unified manager for bilingual text corpus with feature caching and index management"""

    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.source_lines: List[LineObject] = (
            []
        )  # Source text lines (filtered empty lines)
        self.target_lines: List[LineObject] = (
            []
        )  # Target text lines (filtered empty lines)
        self.source_with_empty: List[str] = (
            []
        )  # Source text with empty lines (for output)
        self.target_with_empty: List[str] = (
            []
        )  # Target text with empty lines (for validation)
        self.content_line_map: Dict[int, int] = (
            {}
        )  # Filtered index -> original line number mapping (for source)
        self.target_content_line_map: Dict[int, int] = (
            {}
        )  # Filtered index -> original line number mapping (for target)
        self.line_cache: Dict[str, LineObject] = {}  # Text hash -> LineObject cache

    def load_source(self, file_path: str):
        """Load source file, preserving empty lines for final output"""
        with open(file_path, "r", encoding="utf-8") as f:
            self.source_with_empty = [line.rstrip("\n\r") for line in f]
        filtered_idx = 0
        for orig_idx, line in enumerate(self.source_with_empty):
            if line.strip():  # Non-empty line
                line_obj = self._create_or_get_line(line, is_source=True)
                line_obj.original_line_number = orig_idx + 1
                line_obj.current_index = filtered_idx
                self.source_lines.append(line_obj)
                self.content_line_map[filtered_idx] = orig_idx + 1
                filtered_idx += 1

    def load_target(self, file_path: str):
        """Load target file, filtering empty lines and tracking original line numbers"""
        with open(file_path, "r", encoding="utf-8") as f:
            self.target_with_empty = [line.rstrip("\n\r") for line in f]

        filtered_idx = 0
        for orig_idx, line in enumerate(self.target_with_empty):
            if line.strip():  # Non-empty line
                text = line.strip()
                line_obj = self._create_or_get_line(text, is_source=False)
                line_obj.original_line_number = orig_idx + 1
                line_obj.current_index = filtered_idx
                self.target_lines.append(line_obj)
                self.target_content_line_map[filtered_idx] = orig_idx + 1
                filtered_idx += 1

    def _create_or_get_line(self, text: str, is_source: bool) -> LineObject:
        """Create or retrieve LineObject with caching by text hash"""
        text_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        # If we have a cached prototype for this text, reuse its expensive features
        # (embedding, sentence_count) but return a fresh LineObject instance so
        # mutable metadata like `original_line_number` and `current_index` are
        # not shared across different positions (source vs target, or multiple
        # occurrences). Sharing the same LineObject caused confusing mappings
        # and incorrect original/current indices in logs.
        if text_hash in self.line_cache:
            proto = self.line_cache[text_hash]
            # create a new LineObject instance copying immutable fields
            new_line = LineObject(text=text, is_source=is_source, hash_value=text_hash)
            # copy cached features if present to avoid recomputing
            if proto.sentence_count is not None:
                new_line.sentence_count = int(proto.sentence_count)
            if proto.embedding is not None:
                try:
                    new_line.embedding = proto.embedding.copy()
                except Exception:
                    new_line.embedding = proto.embedding
            return new_line

        # No cached prototype: create, compute features, and cache the prototype
        line = LineObject(text=text, is_source=is_source, hash_value=text_hash)
        self._ensure_features(line)
        # Store a prototype in cache (used only for features copying)
        self.line_cache[text_hash] = line
        return line

    def _ensure_features(self, line: LineObject):
        """Ensure essential features are computed for a line"""
        line.ensure_features(self.text_processor)

    def get_similarity(
        self,
        source_line: LineObject,
        target_line: LineObject,
        use_punctuation_weight: bool = True,
    ) -> float:
        """Compute weighted similarity between normalized embeddings and punctuation patterns

        Args:
            source_line: Source language line
            target_line: Target language line
            use_punctuation_weight: If True, multiply cosine similarity by punctuation similarity weight

        Returns:
            Weighted similarity score (0.0 to 1.0)
        """
        # Compute base cosine similarity between normalized embeddings
        src_norm = source_line.embedding / np.linalg.norm(source_line.embedding)
        tgt_norm = target_line.embedding / np.linalg.norm(target_line.embedding)
        cosine_sim = float(np.dot(src_norm, tgt_norm))

        # If not using punctuation weight, return cosine similarity directly
        if not use_punctuation_weight:
            return cosine_sim

        # Calculate punctuation similarity weight using new formula
        punct_weight = calculate_punctuation_similarity(
            source_line.text, target_line.text
        )

        # Return weighted similarity (cosine * punctuation_weight)
        return cosine_sim * punct_weight
