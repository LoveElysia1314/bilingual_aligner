"""修复数据模型

包含修复相关的所有数据结构、枚举和日志类。

从以下模块迁移：
- bilingual_aligner/corpus.py (RepairLog, RepairType, SplitType, AlignmentState)
- bilingual_aligner/repair_applier.py (PostProcessResult)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import hashlib


class RepairType(Enum):
    """修复操作类型"""

    MERGE_LINES = "MERGE_LINES"
    SPLIT_LINE = "SPLIT_LINE"
    INSERT_LINE = "INSERT_LINE"
    DELETE_LINE = "DELETE_LINE"


class SplitType(Enum):
    """分割类型：区分句子级和短语级分割"""

    HARD_SPLIT = "hard_split"  # 句子级（标点符号）
    SOFT_SPLIT = "soft_split"  # 短语级（逗号、冒号等）


class AlignmentState(Enum):
    """对齐状态"""

    NO_SHIFT = "no_shift"
    SOURCE_AHEAD = "source_ahead"


class LineDifferenceException(Exception):
    """Exception raised when repaired source/target line counts mismatch.

    This specific exception allows callers to distinguish verification
    failures caused by line-count inconsistencies from other ValueError
    conditions.
    """

    def __init__(self, message: str):
        super().__init__(message)


@dataclass
class RepairLog:
    """修复日志条目

    记录单次修复操作的详细信息，包括前后文本、相似度变化等。
    """

    repair_type: RepairType
    description: str
    source_text: str
    target_before: str
    target_after: str
    similarity_before: float
    similarity_after: float
    source_orig_line_numbers: Optional[Tuple[int, ...]] = None  # 原始文件行号
    target_orig_line_numbers: Optional[Tuple[int, ...]] = None  # 原始文件行号
    source_filtered_position: Optional[Tuple[int, ...]] = None  # 内部过滤索引
    target_filtered_position: Optional[Tuple[int, ...]] = None  # 内部过滤索引
    split_type: Optional[SplitType] = None  # 如果是分割操作，记录分割类型
    is_fallback: bool = False  # 如果是备选方案（DELETE/INSERT作为备选）
    timestamp: Optional[datetime] = None  # 修复时间戳

    def __post_init__(self):
        """初始化时间戳"""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "repair_type": self.repair_type.value,
            "description": self.description,
            "source_text": self.source_text,
            "target_before": self.target_before,
            "target_after": self.target_after,
            "similarity_before": self.similarity_before,
            "similarity_after": self.similarity_after,
            "source_orig_line_numbers": self.source_orig_line_numbers,
            "target_orig_line_numbers": self.target_orig_line_numbers,
            "source_filtered_position": self.source_filtered_position,
            "target_filtered_position": self.target_filtered_position,
            "split_type": self.split_type.value if self.split_type else None,
            "is_fallback": self.is_fallback,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class PostProcessResult:
    """后处理结果数据结构"""

    success: bool
    original_target: str
    new_target: str
    repairs_applied: List[RepairLog]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "original_target": self.original_target,
            "new_target": self.new_target,
            "repairs_applied": [r.to_dict() for r in self.repairs_applied],
            "error": self.error,
        }


@dataclass
class RepairOperation:
    """单个修复操作的描述"""

    operation_type: RepairType
    source_indices: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    source_text: str
    quality_score: float

    def __repr__(self):
        return (
            f"RepairOperation({self.operation_type.value}, "
            f"src={self.source_indices}, tgt={self.target_indices}, "
            f"quality={self.quality_score:.3f})"
        )
