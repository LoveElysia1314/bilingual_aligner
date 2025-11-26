# Bilingual Aligner 双语对齐器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python >=3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/bilingual-aligner.svg)](https://pypi.org/project/bilingual-aligner/)

一个高性能的双语文本对齐与修复库，实现了 **优化的单阶段动态规划算法** 和动态惩罚排序。

## 特性

- **优化的单阶段DP算法**: 用单阶段DP + 动态惩罚排序取代了两阶段DP
- **剪枝定理**: 当 `max(调整后分数) > max(未调整分数_剩余部分)` 时提前停止，保证全局最优
 - **最终结构性校验**: 对最终路径检查是否存在连续不同类型的非1:1操作（搜索阶段不再应用动态惩罚）
- **后处理修复**: 对非1:1对齐进行局部修复（拆分、合并、插入）
- **基于嵌入的相似度**: 使用 sentence-transformers 并加权标点符号
- **多语言支持**: 默认模型：Alibaba-NLP/GTE-multilingual-base
- **高性能**: O(nm) 时间复杂度，剪枝带来实际加速

## 安装

```bash
pip install bilingual-aligner
```

安装开发依赖：
```bash
pip install bilingual-aligner[dev]
```

## 快速开始

```python
from bilingual_aligner import TextAligner

# 初始化对齐器
aligner = TextAligner(
    model_name="Alibaba-NLP/GTE-multilingual-base",
    device="cpu"  # 或 "cuda" 使用GPU
)

# 加载双语文本
source_text = "Hello world.\nThis is a test."
target_text = "你好世界。\n这是一个测试。"

# 对齐并修复
aligner.align(source_text, target_text)
aligner.repair()

# 保存结果
aligner.save_results("output/")
aligner.print_report()
```

## 算法概述

### 优化的单阶段DP与动态惩罚排序

算法简化了DP方法：

1. **阶段0：约束节点计算**
   - 硬约束：`ceil(i/2) ≤ j ≤ 2i` 和 `ceil((n-i)/2) ≤ m-j ≤ 2(n-i)`
    - 软约束：通过节点级过滤与操作策略处理（已弃用基于单条边的阈值过滤）
   - 稳定性过滤：前向/后向BFS查找稳定节点

2. **阶段1：单阶段DP与动态惩罚排序**
    - 标准DP计算所有可达路径并按平均相似度评分
    - 路径按未惩罚分数降序枚举和排序
    - 选择后对最终路径执行结构性校验（若检测到连续不同类型的非1:1，则作为异常上报）
    - 通过剪枝定理提前停止，保证效率

### 最终结构性校验

动态惩罚机制已从路径评分阶段移除。当前行为：

- 枚举路径并以平均相似度为主目标进行排序。
- 在选定的最终路径上进行结构性校验，检测前瞻窗口内的不同类型连续非1:1操作；若发现，则将其作为异常报告，供后续处理。

这样既保证了搜索阶段对各种操作类型的公平对待，又能在最终结果中标记结构异常。

### 剪枝定理

**定理**：如果存在 k₀ 使得 `max_adjusted_score(k₀) > max_unadjusted_score`，那么对于所有 k ≥ k₀：
```
max_{i>k₀} adjusted_score(π_i) < max_adjusted_score(k₀)
```

这允许提前停止同时保证全局最优性。

## 配置参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `model_name` | "Alibaba-NLP/GTE-multilingual-base" | 使用的句子嵌入模型名称 |
| `k` | 0.6 | 标点符号权重计算的容差参数 |
| `soft_split_penalty` | 0.05 | 句子间软分割的惩罚 |
| `insert_fallback_score` | 0.6 | 插入回退选项的固定分数 |
| `delete_penalty` | 0.05 | 删除操作的惩罚（未来使用） |
| `consecutive_non_1to1_penalty_max` | (已弃用) | 已弃用：动态惩罚已移除；不再使用 |
| `consecutive_non_1to1_lookahead` | 5 | 最终结构性校验的前瞻步数 |
| `MIN_QUALITY_THRESHOLD` | (已弃用) | 遗留参数；当前单阶段流程中不再使用 |

## API 参考

### TextAligner 类

```python
class TextAligner:
    def __init__(self, source_path, target_path, model_name="Alibaba-NLP/GTE-multilingual-base", **config):
        """初始化对齐器
        
        Args:
            source_path (str): 源文件路径
            target_path (str): 目标文件路径
            model_name (str): 嵌入模型名称
            **config: 其他配置参数
        """
    
    def align(self, source_text, target_text):
        """对齐源文本和目标文本（已弃用，使用repair()）"""
    
    def repair(self):
        """执行修复过程并返回结果"""
    
    def save_results(self, output_dir, **kwargs):
        """保存修复结果"""
    
    def print_report(self):
        """打印修复报告"""
```

#### API 配置参数

通过 `**config` 参数可以传递以下配置：

- `k`: 标点符号权重计算的容差参数 (默认: 0.6)
- `soft_split_penalty`: 句子间软分割的惩罚 (默认: 0.05)
- `insert_fallback_score`: 插入回退选项的固定分数 (默认: 0.6)
- `delete_penalty`: 删除操作的惩罚 (默认: 0.05)
- `consecutive_non_1to1_penalty_max`: 已弃用（动态惩罚已移除）
- `consecutive_non_1to1_lookahead`: 最终结构性校验的前瞻步数 (默认: 5)

### 命令行界面

```bash
# 基本对齐
bilingual-aligner align source.txt target.txt --output output/

# 使用自定义模型
bilingual-aligner align source.txt target.txt --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# 仅修复模式
bilingual-aligner repair alignment.json --output repaired/
```

#### CLI 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--source`, `-s` | - | 源语言文件路径（必需） |
| `--target`, `-t` | - | 目标语言文件路径（必需） |
| `--output`, `-o` | - | 输出目录（必需） |
| `--model` | None | 使用的句子嵌入模型名称 |
| `--k` | 0.8 | 标点符号权重计算的容差参数 |
| `--soft-split-penalty` | 0.05 | 句子间软分割的惩罚 |
| `--insert-fallback-score` | 0.6 | 插入回退选项的固定分数 |
| `--logs-file` | repair_logs.json | 修复日志文件名 |
| `--repaired-file` | <target_basename>_repaired.txt | 修复后目标文件输出名 |
| `--no-texts` | False | 不包含完整源/目标文本在日志中 |
| `--verbose`, `-v` | False | 启用详细输出 |

## 性能

### 时间复杂度
- **阶段0**: O(nm) - 约束计算
- **阶段1**: O(nm + P·L) - DP + 路径枚举（P：路径数，L：路径长度）
- **使用剪枝**: 通常为 O(nm)，提前停止

### 内存使用
- O(nm) 用于DP表
- O(P·L) 用于路径存储

## 高级用法

### 自定义嵌入模型

```python
from sentence_transformers import SentenceTransformer

# 使用自定义模型
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
aligner = TextAligner(model_name=model, device="cuda")
```

### 批量处理

```python
# 处理多个文件
for src_file, tgt_file in file_pairs:
    with open(src_file, 'r', encoding='utf-8') as f:
        source_text = f.read()
    with open(tgt_file, 'r', encoding='utf-8') as f:
        target_text = f.read()
    
    aligner.align(source_text, target_text)
    aligner.repair()
    aligner.save_results(f"output/{src_file.stem}/")
```

### 质量指标

```python
# 访问对齐统计
stats = aligner.get_statistics()
print(f"1:1 比例: {stats['one_to_one_ratio']:.2%}")
print(f"平均相似度: {stats['avg_similarity']:.3f}")
print(f"非1:1操作数: {stats['non_one_to_one_count']}")
```

## 故障排除

### 常见问题

1. **内存使用过高**
   - 减少嵌入计算的批大小
   - 使用更小的嵌入模型
   - 启用剪枝以提前停止

2. **对齐质量差**
    - 检查文本预处理（句子分割、规范化）
    - 尝试不同的嵌入模型
    - 调整节点级过滤（`node_relative_threshold`）或其他阈值参数

3. **性能慢**
   - 如果可用，使用GPU（`device="cuda"`）
   - 启用剪枝（默认启用）
   - 减少文本长度或使用分块处理

### 错误消息

- `"未找到稳定节点"`: 检查文本长度和相似度分数
- `"未找到嵌入模型"`: 验证模型名称或安装 sentence-transformers
- `"索引超出范围"`: 检查文本预处理和行数

## 开发

### 项目结构

```
bilingual_aligner/
├── __init__.py              # 包导出
├── api.py                   # TextAligner 类（高级API）
├── cli.py                   # 命令行界面
├── corpus.py                # 语料库和相似度计算
├── position.py              # LocationRange 和行号映射工具
├── repair_applier.py        # 后处理修复
├── utils.py                 # 工具函数
├── aligners/                # DP 对齐算法实现
│   ├── __init__.py
│   ├── enum_pruning_aligner.py  # v3.0 算法实现
│   └── dp_aligner_two_stage.py  # 遗留 v2.1 实现
└── core/                    # 核心文本处理模块
    ├── __init__.py
    ├── processor.py         # TextProcessor 类
    ├── punctuation.py       # 标点符号处理
    ├── repairer.py          # 主修复协调器
    └── splitter.py          # 通用句子分割器
```

### 运行测试

```bash
# 安装开发依赖
pip install bilingual-aligner[dev]

# 运行测试
pytest tests/
```

### 代码风格

```bash
# 格式化代码
black bilingual_aligner/

# 代码检查
flake8 bilingual_aligner/
```

## 贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加了很棒的功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 引用

如果您在研究中使用了本库，请引用：

```bibtex
@software{bilingual_aligner,
  title = {Bilingual Aligner: 用于文本对齐的优化单阶段DP算法},
  author = {Elysia},
  year = {2024},
  url = {https://github.com/LoveElysia1314/bilingual_aligner}
}
```

## 致谢

- Sentence-transformers 库提供嵌入模型
- Model Context Protocol (MCP) 社区
- bilingual-aligner 项目的贡献者和用户

## 支持

- **问题**: [GitHub Issues](https://github.com/LoveElysia1314/bilingual_aligner/issues)
- **邮箱**: dr.zqr@outlook.com
- **文档**: 查看 [ALGORITHM.md](ALGORITHM.md) 获取详细算法描述
