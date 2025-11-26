# 双语对齐算法：单阶段动态规划

## 目录

1. [问题形式化定义](#问题形式化定义)
2. [数学建模](#数学建模)
3. [算法架构](#算法架构)
4. [约束节点计算](#约束节点计算)
5. [单阶段DP优化](#单阶段dp优化)
6. [后处理修复](#后处理修复)
7. [复杂度分析](#复杂度分析)
8. [实现细节](#实现细节)
9. [参数配置](#参数配置)

---

## 问题形式化定义

### 输入数据结构

给定两个平行的文本序列：
- 源文本：$\text{src}[1..n]$，其中每个元素 $\text{src}[i]$ 是 `LineObject` 实例
- 目标文本：$\text{tgt}[1..m]$，其中每个元素 $\text{tgt}[j]$ 是 `LineObject` 实例

每个 `LineObject` 包含：
- 文本内容
- 词向量嵌入 $\mathbf{v} \in \mathbb{R}^d$
- 标点符号特征
- 原始行号信息

### 对齐图模型

定义对齐图 $G = (V, E)$，其中：
- 顶点集 $V = \{(i,j) : 0 \le i \le n, 0 \le j \le m\}$
- 边集 $E$ 包含三种类型的操作边：
  1. **1:1 操作**（`NO_SHIFT`）：$(i,j) \to (i+1,j+1)$
  2. **2:1 操作**（`SOURCE_AHEAD`）：$(i,j) \to (i+2,j+1)$
  3. **1:2 操作**（`TARGET_SPLIT`）：$(i,j) \to (i+1,j+2)$

### 优化目标

寻找从 $(0,0)$ 到 $(n,m)$ 的路径 $\pi$，最大化调整后得分：

$$\pi^* = \arg\max_{\pi \in \Pi} \left[ \text{score}(\pi) - \text{penalty}(\pi) \right]$$

其中：
- $\text{score}(\pi)$ 是路径的无惩罚总相似度
- $\text{penalty}(\pi)$ 是基于路径结构的动态惩罚

---

## 数学建模

### 相似度函数

对于任意源行集合 $I \subseteq \{1,\ldots,n\}$ 和目标行集合 $J \subseteq \{1,\ldots,m\}$，定义相似度函数：

$$\text{sim}(I,J) = \frac{1}{|I| \times |J|} \sum_{i \in I} \sum_{j \in J} \text{cosine}(\mathbf{v}_i, \mathbf{v}_j) \times w_{\text{punctuation}}(i,j)$$

在代码中，这通过 `BilingualCorpus.get_similarity()` 方法实现，该方法：
1. 计算词向量余弦相似度
2. 应用标点符号权重调整
3. 缓存计算结果以提高性能

### 边权重计算

对于边 $e = ((i,j), (i+\Delta i, j+\Delta j))$，其权重为：

$$w(e) = \text{sim}\left(\{i,\ldots,i+\Delta i-1\}, \{j,\ldots,j+\Delta j-1\}\right)$$

**软约束过滤（已迁移）**：
- 对 1:1 操作（$\Delta i = 1$ 且 $\Delta j = 1$）：无条件接受
- 对非1:1 操作：原先基于单条边阈值的过滤已被弃用，改为在节点级别进行软约束与筛选（参见文档中关于节点级过滤的说明）。

### 路径得分计算

对于路径 $\pi = [e_1, e_2, \ldots, e_k]$，无惩罚总分为：

$$\text{score}(\pi) = \sum_{t=1}^k w(e_t)$$

在代码中，这对应于 `DPAligner._run_stage1_with_penalty_sorting()` 方法中计算的 `total_score`。

---

## 算法架构

### 总体流程

```
输入: src_lines, tgt_lines (List[LineObject])
    ↓
[约束计算] 计算稳定可达节点集合 S*
    ↓
[单阶段DP] 最大化平均相似度路径
    ├── 预计算句子嵌入
    ├── 计算边权重（余弦相似度）
    ├── DP寻找最大平均相似度路径
    └── 提取对齐步骤
    ↓
[后处理] 修复应用与异常检测
    ├── 识别非1:1对齐位置
    ├── 生成修复候选（拆分/合并/插入）
    ├── 排序并应用修复
    ├── 检测对齐异常（相邻不同类型非1:1操作）
    └── 验证修复结果
    ↓
输出: 对齐路径、修复日志与异常报告
```

### 代码中的主要类

1. **`DPAligner`**（`enum_pruning_aligner.py`）：核心对齐算法
2. **`BilingualCorpus`**（`corpus.py`）：语料库管理与相似度计算
3. **`RepairApplier`**（`repair_applier.py`）：后处理修复
4. **`TextAligner`**（`core/repairer.py`）：主协调器

---

## 阶段0：约束节点计算

### 硬约束定义

基于文本长度比例假设，定义硬约束区域：

$$S_{\text{hard}} = \left\{ (i,j) : 
\begin{cases}
\left\lceil \frac{i}{2} \right\rceil \le j \le 2i \\
\left\lceil \frac{n-i}{2} \right\rceil \le m-j \le 2(n-i)
\end{cases}
\right\}$$

**数学解释**：
- 前半约束：源文本前 $i$ 行对应目标文本约 $i$ 行，允许 ±50% 的长度变化
- 后半约束：剩余部分保持相同的比例关系

### 可达性分析

1. **前向可达集** $R_{\text{forward}}$：从 $(0,0)$ 出发，沿三种允许操作可达的所有节点
2. **后向可达集** $R_{\text{backward}}$：从 $(n,m)$ 反向可达的所有节点

### 稳定节点集合

最终稳定节点集合为三者的交集：

$$S^* = S_{\text{hard}} \cap R_{\text{forward}} \cap R_{\text{backward}}$$

在代码中，这通过 `_compute_stable_nodes()` 方法实现，显著减少搜索空间。

---

## 单阶段DP优化

### 优化目标

寻找从 $(0,0)$ 到 $(n,m)$ 的路径 $\pi$，最大化平均相似度：

$$\pi^* = \arg\max_{\pi \in \Pi} \frac{\sum_{e \in \pi} w(e)}{|\pi|}$$

其中 $w(e)$ 是边的相似度权重（余弦相似度）。

### DP计算

对于每个节点 $(i,j) \in S^*$，计算最大累积分数：

$$\text{dp}[i][j] = \max \begin{cases}
\text{dp}[i-1][j-1] + w((i-1,j-1) \to (i,j)) & \text{(1:1)} \\
\text{dp}[i-2][j-1] + w((i-2,j-1) \to (i,j)) & \text{(2:1)} \\
\text{dp}[i-1][j-2] + w((i-1,j-2) \to (i,j)) & \text{(1:2)}
\end{cases}$$

边界条件：$\text{dp}[0][0] = 0$

### 路径重建

从终点 $(n,m)$ 反向遍历父指针，重建最优路径。

### 平均相似度计算

$$\text{avg\_similarity} = \frac{\text{dp}[n][m]}{\text{number\_of\_operations}}$$

如果平均相似度低于阈值 $T_{\min}$，记录警告。

---

$$\text{penalty}(\pi) = \sum_{\text{符合条件的操作对}} \text{penalty}(d)$$

在代码中，这通过 `_compute_path_penalty()` 方法实现。

---

## 后处理修复

### 修复目标

将DP识别出的非1:1对齐位置转换为1:1对齐，通过局部文本操作：
- **2:1操作** → 源行拆分或目标行插入
- **1:2操作** → 目标行合并

### 修复候选生成

#### 2:1操作（SOURCE_AHEAD）
1. **硬拆分候选**：在标点边界处拆分源行
2. **软拆分候选**：在非标点位置拆分，基于语义相似度
3. **插入候选**（保底）：在目标侧插入占位行

拆分得分计算：
$$\text{split\_score} = \frac{\text{sim}(I_1, J) + \text{sim}(I_2, J)}{2}$$
其中 $I_1 \cup I_2 = \{i, i+1\}$ 是源行的拆分。

#### 1:2操作（TARGET_SPLIT）
- **合并候选**：将两个目标行合并为单行
- 强制执行，不计算得分

### 修复应用策略

1. **降序排序**：按目标索引降序处理修复候选，避免索引漂移
2. **增量更新**：每次修复后立即重建索引映射
3. **验证检查**：确保索引连续性和映射一致性

### 占位行处理

插入的占位行标记为 synthetic（`original_line_number = -1`），不参与后续的全局相似度计算，避免污染语义空间。

### 异常检测

在最终对齐结果中检测相邻的不同类型非1:1操作，并报告为异常：
- 连续的SOURCE_AHEAD和TARGET_SPLIT操作被视为潜在问题
- 异常信息记录在修复日志的`state.exceptions`中

---

## 复杂度分析

### 时间复杂度

#### 约束计算
- 硬约束枚举：$O(nm)$
- 前向/后向BFS：$O(|S^*|) \le O(nm)$
- **总计**：$O(nm)$

#### 单阶段DP
- DP计算：$O(|S^*|) \le O(nm)$
- 路径重建：$O(\max(n,m))$

#### 后处理修复
- 修复候选生成：$O(\text{\#非1:1操作} \times \text{候选数})$
- 修复应用：$O(\text{\#修复} \times \log \text{\#修复})$（排序成本）

#### 总体复杂度
- **最坏情况**：$O(nm)$
- **实际表现**：接近 $O(nm)$

### 空间复杂度

- DP表与父指针：$O(|S^*|) \le O(nm)$
- 边权重缓存：$O(|E|) \le O(3|S^*|)$
- **总计**：$O(nm)$

---

## 参数配置

### 核心参数定义

在 `DPAligner` 类中定义了以下关键参数：

```python
# Configuration parameters
MIN_QUALITY_THRESHOLD = 0.75  # T_min: minimum average similarity
```

### 参数详细说明

#### 最小质量阈值（`MIN_QUALITY_THRESHOLD`）

**数学意义**：$T_{\min} = 0.75$

**作用**：定义可接受对齐的最小平均相似度阈值。

**在算法中的角色**：
- 用于质量验证和日志记录
- 如果最终路径的平均相似度低于此阈值，记录警告

### 参数配置接口

用户可以通过配置字典自定义这些参数：

```python
config = {
    "min_quality_threshold": 0.75,  # 默认值
}
```
    "consecutive_non_1to1_lookahead": 5,      # 默认值
    "min_quality_threshold": 0.75,            # 默认值
}

aligner = DPAligner(corpus, config)
```

### 参数调优建议

#### 针对不同语言对的调整

1. **高相似度语言对**（如英文-法文）：
    - 可适当提高节点级过滤阈值（如 `node_relative_threshold`），以减少非必要的非1:1候选
    - 降低对非1:1操作的容忍度

2. **低相似度语言对**（如英文-中文）：
    - 可适当降低节点级过滤阈值（如 `node_relative_threshold`），以允许更多候选
    - 增加对非1:1操作的接受度

3. **结构差异大的语言对**：
   - 可增加 `consecutive_non_1to1_lookahead`（如 7-10）
   - 更严格地检测长距离的病理性对齐模式

#### 性能与质量权衡

- **追求最高质量**：使用默认参数，算法已优化为在质量和计算效率间取得平衡
- **追求最快速度**：可提高节点级过滤阈值（如 `node_relative_threshold`）或减少 `consecutive_non_1to1_lookahead` 来减少候选边和路径枚举。
- **处理特殊文本**：对于诗歌、代码等特殊文本，可能需要调整参数以适应其结构特性

### 默认参数的科学依据

当前默认参数基于大量双语文本对齐实验得出：

1. **边级阈值已弃用**：原先的 $\theta = 0.5$（基于单边过滤）已迁移为节点级过滤策略（例如 `node_relative_threshold`）以提高稳定性
2. **$P_{\max} = 1.0$**：足够严厉地惩罚病理性对齐，同时不过度影响孤立的高质量非1:1
3. **$L = 5$**：覆盖大多数实际中的连续非1:1模式
4. **$T_{\min} = 0.75$**：保证对齐质量的基本要求

这些参数共同作用，实现了v3.0算法的核心优势：在保持高质量对齐的同时，通过动态惩罚机制灵活处理非1:1操作，避免刚性规则导致的次优解。

---

## 实现细节

### 数据结构映射

#### 对齐步骤表示
```python
AlignmentStep = {
    'src_start': int,      # 源起始索引（包含）
    'src_end': int,        # 源结束索引（包含）
    'tgt_start': int,      # 目标起始索引（包含）
    'tgt_end': int,        # 目标结束索引（包含）
    'score': float,        # 步骤相似度
    'operation': Enum,     # NO_SHIFT, SOURCE_AHEAD, TARGET_SPLIT
}
```

#### DP状态表示
```python
DPState = {
    'total_score': float,      # 到达该节点的最大总分
    'parent': Optional[Tuple], # 父节点坐标
    'parent_op': Optional[Enum], # 父操作类型
}
```

### 关键算法方法

#### 1. 稳定节点计算（`_compute_stable_nodes`）
```python
def _compute_stable_nodes(self, n, m):
    # 1. 硬约束过滤
    hard_nodes = {(i,j) for i in range(n+1) for j in range(m+1) 
                  if self._satisfies_hard_constraints(i, j, n, m)}
    
    # 2. 前向可达性
    forward_reachable = self._bfs_forward(hard_nodes)
    
    # 3. 后向可达性  
    backward_reachable = self._bfs_backward(hard_nodes)
    
    # 4. 取交集
    return hard_nodes & forward_reachable & backward_reachable
```

#### 2. 动态惩罚计算（`_compute_path_penalty`）
```python
def _compute_path_penalty(self, path):
    total_penalty = 0.0
    last_non_1to1_type = None
    last_non_1to1_pos = None
    
    for step in path:
        if step.operation != NO_SHIFT:
            current_type = (step.src_end - step.src_start + 1,
                           step.tgt_end - step.tgt_start + 1)
            current_pos = (step.src_end, step.tgt_end)
            
            if (last_non_1to1_type is not None and 
                current_type != last_non_1to1_type):
                distance = self._manhattan_distance(last_non_1to1_pos, current_pos)
                if distance <= self.lookahead:
                    penalty = max(0, self.penalty_max - 0.2 * distance)
                    total_penalty += penalty
            
            last_non_1to1_type = current_type
            last_non_1to1_pos = current_pos
    
    return total_penalty
```

#### 3. 路径枚举与排序（`_run_stage1_with_penalty_sorting`）
```python
def _run_stage1_with_penalty_sorting(self, stable_nodes, edge_weights):
    # 1. 标准DP（无惩罚）
    dp_table, parent_table = self._run_standard_dp(stable_nodes, edge_weights)
    
    # 2. 路径枚举
    all_paths = self._enumerate_paths(dp_table, parent_table)
    
    # 3. 按无惩罚总分降序排序
    sorted_paths = sorted(all_paths, key=lambda p: p.total_score, reverse=True)
    
    # 4. 动态惩罚应用与最优追踪
    best_adjusted_score = -float('inf')
    best_path = None
    
    for i, path in enumerate(sorted_paths):
        # 计算动态惩罚
        penalty = self._compute_path_penalty(path.steps)
        adjusted_score = path.total_score - penalty
        
        # 更新最优
        if adjusted_score > best_adjusted_score:
            best_adjusted_score = adjusted_score
            best_path = path
        
        # 修剪定理提前停止检查
        if i < len(sorted_paths) - 1:
            max_remaining = sorted_paths[i+1].total_score
            if best_adjusted_score > max_remaining:
                break  # 提前停止
    
    return best_path
```

### 4. 修复应用（`RepairApplier.apply_repairs`）
```python
def apply_repairs(self, corpus, alignment_steps):
    # 1. 识别非1:1操作并生成修复候选
    repair_candidates = []
    for step in alignment_steps:
        if step.operation == SOURCE_AHEAD:
            candidates = self._generate_split_candidates(corpus, step)
            repair_candidates.extend(candidates)
        elif step.operation == TARGET_SPLIT:
            candidate = self._generate_merge_candidate(corpus, step)
            repair_candidates.append(candidate)
    
    # 2. 按目标索引降序排序
    repair_candidates.sort(key=lambda c: c.tgt_start, reverse=True)
    
    # 3. 逐一应用修复
    for candidate in repair_candidates:
        corpus = self._apply_single_repair(corpus, candidate)
        # 重建索引映射
        corpus.rebuild_index_mapping()
    
    # 4. 验证修复结果
    self._validate_repairs(corpus)
    
    return corpus
```
