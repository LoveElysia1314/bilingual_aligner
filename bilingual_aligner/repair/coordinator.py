"""修复协调器

负责决策修复策略，评估对齐质量，选择最佳修复方案。
从 core/repairer.py 中提取，职责限制为决策逻辑。
"""

from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import logging

from .models import (
    RepairLog,
    RepairType,
    RepairOperation,
    PostProcessResult,
    SplitType,
    AlignmentState,
)


class RepairStrategy(Enum):
    """修复策略"""

    MERGE_STRATEGY = "merge"  # 合并行对齐
    SPLIT_STRATEGY = "split"  # 分割行对齐
    INSERT_STRATEGY = "insert"  # 插入行
    NO_REPAIR = "no_repair"  # 不修复


class RepairCoordinator:
    """修复协调器

    职责：
    - 决定哪些对齐需要修复
    - 选择最佳的修复策略
    - 评估修复质量
    - 评估对齐的可靠性

    不负责：
    - 实际执行修复操作（由 RepairApplier 执行）
    - 管理文本数据（由 Processor 管理）
    """

    def __init__(self, similarity_threshold: float = 0.7, processor=None):
        """初始化修复协调器

        Args:
            similarity_threshold: 相似度阈值
            processor: 文本处理器
        """
        self.similarity_threshold = similarity_threshold
        self.processor = processor
        self.logger = logging.getLogger(__name__)

    def decide_repairs(
        self,
        alignments: List[Dict[str, Any]],
        source_lines: List[str],
        target_lines: List[str],
        alignment_probs: Optional[List[float]] = None,
    ) -> List[RepairOperation]:
        """决定需要进行哪些修复

        Args:
            alignments: 对齐结果列表
            source_lines: 源文本行列表
            target_lines: 目标文本行列表
            alignment_probs: 对齐概率列表

        Returns:
            修复操作列表
        """
        repairs = []

        for i, alignment in enumerate(alignments):
            if not self._is_reliable_alignment(alignment, alignment_probs, i):
                strategy = self._select_repair_strategy(
                    alignment, source_lines, target_lines
                )
                if strategy != RepairStrategy.NO_REPAIR:
                    repair_op = RepairOperation(
                        strategy=strategy.value,
                        alignment_index=i,
                        source_indices=alignment.get("source_indices", ()),
                        target_indices=alignment.get("target_indices", ()),
                        confidence=alignment.get("similarity", 0.0),
                    )
                    repairs.append(repair_op)

        return repairs

    def _is_reliable_alignment(
        self,
        alignment: Dict[str, Any],
        alignment_probs: Optional[List[float]] = None,
        index: int = 0,
    ) -> bool:
        """判断对齐是否可靠

        Args:
            alignment: 对齐结果
            alignment_probs: 对齐概率列表
            index: 对齐索引

        Returns:
            True 如果对齐可靠，False 否则
        """
        # 检查相似度
        similarity = alignment.get("similarity", 0.0)
        if similarity < self.similarity_threshold:
            return False

        # 检查对齐概率
        if alignment_probs and index < len(alignment_probs):
            prob = alignment_probs[index]
            if prob < 0.5:  # 概率阈值
                return False

        return True

    def _select_repair_strategy(
        self,
        alignment: Dict[str, Any],
        source_lines: List[str],
        target_lines: List[str],
    ) -> RepairStrategy:
        """选择最佳修复策略

        Args:
            alignment: 对齐结果
            source_lines: 源文本行列表
            target_lines: 目标文本行列表

        Returns:
            修复策略
        """
        source_indices = alignment.get("source_indices", ())
        target_indices = alignment.get("target_indices", ())
        alignment_state = alignment.get("alignment_state", AlignmentState.NO_SHIFT)

        # 多对一对齐：目标行数多于源行数 → 分割
        if len(target_indices) > len(source_indices):
            return RepairStrategy.SPLIT_STRATEGY

        # 一对多对齐：源行数多于目标行数 → 合并
        if len(source_indices) > len(target_indices):
            return RepairStrategy.MERGE_STRATEGY

        # 偏移对齐：源提前 → 插入
        if alignment_state == AlignmentState.SOURCE_AHEAD:
            return RepairStrategy.INSERT_STRATEGY

        # 默认不修复
        return RepairStrategy.NO_REPAIR
