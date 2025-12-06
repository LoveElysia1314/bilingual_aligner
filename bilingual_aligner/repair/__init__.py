"""修复模块

管理对齐修复的完整流程：
- models: 数据结构和枚举
- executor: 修复执行（核心修复逻辑）
- coordinator: 修复策略决策
"""

from .models import (
    RepairType,
    SplitType,
    AlignmentState,
    RepairLog,
    PostProcessResult,
    RepairOperation,
)
from .executor import RepairExecutor
from .coordinator import RepairCoordinator, RepairStrategy

# 向后兼容别名
RepairApplier = RepairExecutor

__all__ = [
    "RepairType",
    "SplitType",
    "AlignmentState",
    "RepairLog",
    "PostProcessResult",
    "RepairOperation",
    "RepairExecutor",
    "RepairApplier",
    "RepairCoordinator",
    "RepairStrategy",
]
