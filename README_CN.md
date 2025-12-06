[English](README.md) | 中文

# Bilingual Aligner 双语对齐器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python >=3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/bilingual-aligner.svg)](https://pypi.org/project/bilingual-aligner/)

一个高性能的双语文本对齐与修复库，实现了**优化的单阶段动态规划算法**和节点级软约束。

## 核心特性

- **单阶段动态规划**：最大化总相似度（边权重之和），公平对待所有操作类型（1:1、2:1、1:2）；平均相似度在提取路径后作为报告指标。
- **节点级软约束**：通过相对阈值动态过滤低质量节点，提高搜索效率
- **结构性校验**：对最终路径检测连续不同类型的非1:1操作，并报告为异常
- **后处理修复**：针对非1:1对齐进行局部修复（拆分、合并、插入）
- **基于嵌入的相似度**：使用 sentence-transformers 计算语义相似度
- **多语言支持**：默认模型 Alibaba-NLP/GTE-multilingual-base，支持 100+ 种语言
- **高性能**：$O(nm)$ 时间复杂度，硬/软约束剪枝显著加速

## 快速开始

### 安装

```bash
pip install bilingual-aligner
```

或安装开发版本：

```bash
pip install bilingual-aligner[dev]
```

### 基本使用

```python
from bilingual_aligner import TextAligner

# 初始化对齐器
aligner = TextAligner(
    source_path="source.txt",
    target_path="target.txt",
    model_name="Alibaba-NLP/GTE-multilingual-base",
    device="cpu"  # 或 "cuda"
)

# 执行对齐与修复
result = aligner.repair()

# 保存结果
aligner.save_results(result, output_dir="output/")

# 打印报告
aligner.print_report(result, output_dir="output/")
```

### 命令行使用

```bash
# 基本对齐
bilingual-aligner align source.txt target.txt --output output/

# 指定模型
bilingual-aligner align source.txt target.txt \
  --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --output output/

# 自定义参数
bilingual-aligner align source.txt target.txt \
  --output output/ \
  --node-relative-threshold 0.8 \
  --consecutive-non-1to1-lookahead 7
```

## 算法概述

### 三阶段流程

#### 阶段 0：约束节点计算

从搜索空间的 $O(n \times m)$ 个节点中，通过多层过滤识别稳定的可达节点集合 $S^*$：

1. **硬约束过滤**：基于文本长度比例 $\lceil i/2 \rceil \le j \le 2i$，定义基本的可行域
2. **节点级软约束**：对于每个节点，评估其最佳输出操作的相似度是否超过 $\theta \times \text{element\_best}$
    - $\theta$ 为相对阈值（默认 0.8）
    - `element_best` 的计算采用保守策略：当源端和译端的1:1行级最佳值均可用时，取两者中的较小值（即 `min(src_best[i], tgt_best[j])`）作为基准；若仅一侧可用，则使用该侧的最佳值。
3. **可达性分析**：前向BFS确保节点可从起点到达，后向BFS确保可到达终点
4. **迭代收敛**：重复应用可达性过滤直至稳定

**效果**：通常将 $O(nm)$ 个节点减少到数百至数千个，显著加速后续计算。

-#### 阶段 1：单阶段动态规划

直接在稳定节点集上进行DP，最大化总相似度（边权重之和）：

- **目标函数**：$\max_{\pi} \sum w_e$

注：实现的DP以总相似度为搜索目标，平均相似度 $\text{avg\_sim}(\pi)=\text{score}(\pi)/|\pi|$ 在路径提取后计算并用于报告与质量评估。
- **转移方程**：对每个节点 $(i,j)$，尝试从三种可能的前驱节点（对应1:1、2:1、1:2操作）转移，取最大累积分数
- **路径重建**：从终点反向追踪父指针，得到最优节点序列
- **操作提取**：相邻节点对的位置差异确定每一步的操作类型和相似度分数

**保证**：该DP在所有满足硬/软约束和可达性条件的路径中找到最优解，且找到的路径一定从 $(0,0)$ 到 $(n,m)$。

#### 阶段 2：最终结构性校验

对最终得到的对齐路径进行结构性检查，而**不在搜索阶段应用动态惩罚**：

- **目的**：识别可能的病理性对齐模式（相邻的不同类型非1:1操作）
- **规则**：若检测到距离不超过 $L$（前瞻窗口，默认5）的不同类型非1:1操作对，标记为异常
- **报告**：异常信息保存在修复日志中，供后续分析或手动审核

这一设计确保搜索阶段的公平性和质量，同时能够识别结构问题。

### 后处理修复

对DP识别的非1:1操作进行局部修复，尝试转换为1:1对齐：

**2:1操作处理**（源行领先）：
- 尝试在源行的标点边界处进行硬拆分
- 尝试基于语义相似度的软拆分（选择使合并分数最高的拆分位置）
- 如上述均失败，插入占位行（synthetic）到目标文本

**1:2操作处理**（目标行拆分）：
- 直接合并两个目标行为单行

**应用策略**：
- 按目标行索引降序处理，避免修复过程中的索引漂移
- 每次修复后重建索引映射，确保一致性
- 修复后的占位行不参与全局相似度计算，防止语义污染

自动重试以修正索引偏移异常：

- 修复完成后，流水线会检测修复结果中是否存在连续的邻近索引异常（例如多行出现 `TARGET_BEFORE` 或 `TARGET_AFTER`），若发现则自动重试完整的对齐+修复流程，并逐步放宽 `node_relative_threshold` 以扩大搜索空间。
- 放宽策略：依次按累计放宽量 0.05、0.15、0.30 应用（即分别在原始阈值上减去 0.05、0.15、0.30），阈值变得更小表示节点过滤更宽松。
- 仅返回最终（成功或最后一次）修复日志；中间尝试不会写入持久化日志以避免混淆。

## 配置参数

### 主要参数

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `model_name` | `"Alibaba-NLP/GTE-multilingual-base"` | str | 句子嵌入模型名称 |
| `device` | `"cpu"` | str | 计算设备（`"cpu"` 或 `"cuda"`） |
| `node_relative_threshold` | `0.8` | float | 节点级软约束的相对阈值 $\theta$ |
| `consecutive_non_1to1_lookahead` | `5` | int | 结构性校验的前瞻窗口 $L$ |
| `soft_split_penalty` | `0.05` | float | 句子间软分割的惩罚 |
| `insert_fallback_score` | `0.6` | float | 插入占位行的固定相似度 |
| `delete_penalty` | `0.05` | float | 删除操作的惩罚（预留） |

### 参数调优

#### 针对不同语言对

1. **高相似度语言对**（英文-法文、中文-日文等）
   - 建议：增加 `node_relative_threshold` 至 0.85-0.95
   - 效果：搜索空间更严格，找到更多1:1对齐

2. **低相似度语言对**（英文-中文、英文-阿拉伯文等）
   - 建议：降低 `node_relative_threshold` 至 0.5-0.65
   - 效果：搜索空间更宽松，容纳更多结构差异

3. **结构差异大的语言对**
   - 建议：增加 `consecutive_non_1to1_lookahead` 至 7-10
   - 效果：更严格地检测病理性模式

#### 性能与质量权衡

- **高质量模式**：使用默认参数，已在多种语言对和文本类型上验证
- **高速模式**：增加 `node_relative_threshold` 或减少 `consecutive_non_1to1_lookahead`
- **特殊文本**：诗歌、代码等可能需要调整 `node_relative_threshold`

## API 参考

### TextAligner 类

```python
class TextAligner:
    def __init__(
        self,
        source_path: str,
        target_path: str,
        model_name: str = "Alibaba-NLP/GTE-multilingual-base",
        text_processor: Optional[TextProcessor] = None,
        **config
    )
    
    def repair() -> Dict[str, Any]
        """执行修复流程并返回结果字典"""
    
    def save_results(
        self,
        result: Dict[str, Any],
        output_dir: Optional[str] = None,
        logs_file: Optional[str] = None,
        repaired_file: Optional[str] = None,
        include_texts: bool = True
    )
        """保存修复结果到输出目录"""
    
    def print_report(
        self,
        result: Dict[str, Any],
        output_dir: Optional[str] = None,
        show_sample_logs: int = 3
    )
        """打印修复报告"""
```

### 返回结果结构

`repair()` 返回的结果字典包含：

```
{
    "repaired_lines": [str],  # 修复后的目标文本行
    "repair_logs": [RepairLog],  # 详细的修复日志
    "line_number_mapping": LineNumberMapping,  # 原始/修复后行号映射
    "stats": {  # 统计信息
        "total_repairs": int,
        "source_content_lines": int,
        "target_content_lines_before": int,
        "target_content_lines_after": int,
        "line_difference": int,
        "similarity_improvement": float,
        "dp_time_seconds": float,
        "postprocess_time_seconds": float,
        "total_time_seconds": float,
        "dp_operations": {  # DP的操作统计
            "NO_SHIFT": int,
            "SOURCE_AHEAD": int,
            "TARGET_SPLIT": int
        },
        "similarity_statistics": {  # 相似度统计
            "min": float,
            "max": float,
            "mean": float,
            "median": float,
            "std": float,
            "low_quality_count": int
        }
    },
    "state": {
        "exceptions": [str]  # 检测到的异常列表
    }
}
```

## 性能指标

### 时间复杂度

- **阶段0（约束计算）**：$O(nm)$
- **阶段1（DP）**：$O(nm)$
- **阶段2（校验）**：$O(|\pi|) = O(\max(n,m))$
- **后处理**：$O(\text{修复数} \cdot \log \text{修复数})$
- **总计**：$O(nm + \text{修复数} \cdot \log \text{修复数}) \approx O(nm)$

### 空间复杂度

- **DP表与缓存**：$O(nm)$
- **修复日志与结果**：$O(n+m)$
- **总计**：$O(nm)$

### 实测性能

在典型应用场景（源文本1000行、目标文本1000行）中：

- **单阶段DP**：通常 < 5 秒（CPU）
- **后处理修复**：通常 < 2 秒
- **总耗时**：< 10 秒（取决于模型加载时间）

## 质量指标

### 输出质量评估

修复完成后可查看以下指标：

- **1:1对齐比例**：最终1:1操作占总操作的百分比
- **平均相似度**：对齐路径的平均相似度分数（0-1）
- **修复率**：修复操作数占源文本行数的百分比
- **异常数**：检测到的结构性异常数量
- **低质量对齐数**：相似度 < 0.6 的对齐数量

### 质量判断指南

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| 1:1比例 | > 95% | 80-95% | < 80% |
| 平均相似度 | > 0.85 | 0.70-0.85 | < 0.70 |
| 修复率 | < 2% | 2-5% | > 5% |
| 异常数 | 0 | 1-2 | > 2 |

## 故障排除

### 常见问题

#### 1. 内存使用过高
**症状**：执行时内存消耗过大或OOM错误
- 减少文本长度或分块处理
- 使用更小的嵌入模型
- 增加 `node_relative_threshold` 以减少节点数

#### 2. 对齐质量差
**症状**：平均相似度低，大量低质量对齐
- 检查文本预处理（句子分割是否正确）
- 尝试调整 `node_relative_threshold`（较低值允许更多探索）
- 尝试不同的嵌入模型
- 检查源文本和目标文本是否确实平行

#### 3. 性能慢
**症状**：DP阶段耗时较长
- 如可用，使用GPU（`device="cuda"`）
- 增加 `node_relative_threshold` 以减少节点数
- 考虑使用更小/更快的嵌入模型

#### 4. 修复失败或异常过多
**症状**：修复阶段异常，或异常报告过多
- 检查源/目标文本行数差异是否过大（> 50%）
- 调整 `consecutive_non_1to1_lookahead` 以改变异常检测灵敏度
- 考虑文本是否真正平行

### 错误消息解释

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `FileNotFoundError: Source file not found` | 源文件路径不存在 | 检查文件路径 |
| `No stable nodes found` | 约束太严格，无可行节点 | 降低 `node_relative_threshold` |
| `LineDifferenceException` | 修复后源目行数不匹配 | 检查修复逻辑或输入文本 |
| `RepairRateException` | 修复率超过5% | 检查文本质量或调整参数 |
| `LowSimilarityException` | 多个低相似度对齐 | 检查语言对或模型选择 |

## 高级用法

### 自定义嵌入模型

```python
from sentence_transformers import SentenceTransformer

aligner = TextAligner(
    source_path="source.txt",
    target_path="target.txt",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
```

### 参数自定义

```python
aligner = TextAligner(
    source_path="source.txt",
    target_path="target.txt",
    node_relative_threshold=0.8,
    consecutive_non_1to1_lookahead=7,
    device="cuda"
)
```

### 批量处理

```python
from pathlib import Path

file_pairs = [
    ("source1.txt", "target1.txt"),
    ("source2.txt", "target2.txt"),
]

for src, tgt in file_pairs:
    aligner = TextAligner(src, tgt)
    result = aligner.repair()
    aligner.save_results(result, output_dir=f"output/{Path(src).stem}/")
```

### 访问详细修复日志

```python
result = aligner.repair()

# 查看所有修复
for repair_log in result["repair_logs"]:
    print(f"修复类型: {repair_log.repair_type}")
    print(f"相似度改进: {repair_log.similarity_after - repair_log.similarity_before:.3f}")
    print(f"源行范围: {repair_log.src_start}-{repair_log.src_end}")
    print(f"目标行范围: {repair_log.tgt_start}-{repair_log.tgt_end}")
    print()

# 查看统计信息
stats = result["stats"]
print(f"1:1操作数: {stats['dp_operations']['NO_SHIFT']}")
print(f"总修复数: {stats['total_repairs']}")
print(f"平均相似度: {stats['similarity_statistics']['mean']:.3f}")
print(f"异常数: {len(result['state']['exceptions'])}")
```

## 开发与贡献

### 项目结构

```
bilingual_aligner/
├── __init__.py
├── api.py                    # 高级API
├── cli.py                    # 命令行界面
├── corpus.py                 # 语料库与相似度计算
├── position.py               # 行号映射工具
├── utils.py                  # 工具函数
├── core/
│   ├── processor.py          # 嵌入与编码
│   ├── splitter.py           # 句子分割
│   ├── punctuation.py        # 标点符号处理
│   └── repairer.py           # 修复协调器
├── alignment/
│   ├── base.py               # 基类
│   └── enum_aligner.py       # 单阶段DP算法
├── repair/
│   ├── models.py             # 数据结构
│   ├── executor.py           # 修复执行器
│   └── coordinator.py        # 修复协调器
└── analyzer/                 # 分析工具
    ├── base.py
    ├── similarity.py
    ├── encoding.py
    ├── punctuation.py
    └── comparison.py
```

### 运行测试

```bash
pip install bilingual-aligner[dev]
pytest tests/ -v
```

### 代码风格

```bash
black bilingual_aligner/
flake8 bilingual_aligner/
```

## 引用

如果您在研究中使用了本库，请引用：

```bibtex
@software{bilingual_aligner,
  title = {Bilingual Aligner: 优化的单阶段动态规划双语对齐算法},
  author = {Elysia},
  year = {2024},
  url = {https://github.com/LoveElysia1314/bilingual_aligner}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 支持

- **问题反馈**：[GitHub Issues](https://github.com/LoveElysia1314/bilingual_aligner/issues)
- **电子邮件**：dr.zqr@outlook.com
- **详细文档**：[ALGORITHM.md](ALGORITHM.md) 获取算法细节

## 致谢

- [Sentence Transformers](https://www.sbert.net/) - 句子嵌入模型库
- [Alibaba NLP](https://huggingface.co/Alibaba-NLP) - 预训练多语言模型
- 所有贡献者和用户的反馈与支持

