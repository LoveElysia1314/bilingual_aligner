[中文](README_CN.md) | English

下面是 README_CN_NEW.md 的英文翻译：

# Bilingual Aligner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python >=3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![PyPI version](https://badge.fury.io/py/bilingual-aligner.svg)](https://pypi.org/project/bilingual-aligner/)

A high-performance bilingual text alignment and repair library that implements an optimized single-stage dynamic programming algorithm and node-level soft constraints.

## Key Features

- Single-stage dynamic programming: directly maximizes average similarity and treats all operation types fairly (1:1, 2:1, 1:2).  
 - Single-stage dynamic programming: maximizes total similarity (sum of edge weights) while treating all operation types fairly (1:1, 2:1, 1:2). Average similarity is computed afterwards for reporting.  
- Node-level soft constraints: dynamically filter low-quality nodes with a relative threshold to improve search efficiency.  
- Structural validation: detect consecutive different non-1:1 operations in the final path and report them as exceptions.  
- Post-processing repairs: local repairs for non-1:1 alignments (split, merge, insert).  
- Embedding-based similarity: uses sentence-transformers for semantic similarity.  
- Multilingual support: default model `Alibaba-NLP/GTE-multilingual-base`, supports 100+ languages.  
- High performance: O(nm) time complexity; pruning by hard/soft constraints offers significant acceleration.

## Quick Start

### Installation

```bash
pip install bilingual-aligner
```

Or install development extras:

```bash
pip install bilingual-aligner[dev]
```

### Basic Usage

```python
from bilingual_aligner import TextAligner

# Initialize the aligner
aligner = TextAligner(
    source_path="source.txt",
    target_path="target.txt",
    model_name="Alibaba-NLP/GTE-multilingual-base",
    device="cpu"  # or "cuda"
)

# Run alignment and repair
result = aligner.repair()

# Save results
aligner.save_results(result, output_dir="output/")

# Print report
aligner.print_report(result, output_dir="output/")
```

### CLI Usage

```bash
# Basic alignment
bilingual-aligner align source.txt target.txt --output output/

# Specify model
bilingual-aligner align source.txt target.txt \
  --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --output output/

# Custom parameters
bilingual-aligner align source.txt target.txt \
  --output output/ \
  --node-relative-threshold 0.8 \
  --consecutive-non-1to1-lookahead 7
```

## Algorithm Overview

### Three-stage Flow

#### Stage 0: Constraint Node Computation

From the O(n × m) search space, identify the stable reachable node set S* via multi-layer filtering:

1. Hard-constraint filtering: define the feasible domain by text-length proportions: ⌈i/2⌉ ≤ j ≤ 2i.  
2. Node-level soft constraint: for each node, evaluate whether its best outgoing operation
    similarity exceeds θ × element_best
    - θ is the relative threshold (default 0.8).
    - element_best is computed from per-line 1:1 best scores on both sides: when both
      source and target sides are available, we use the smaller of the two per-line bests
      (i.e. element_best = min(src_best[i], tgt_best[j])) to be conservative on
      heterogeneous pairs; if only one side is available, that side's best is used.
3. Reachability analysis: forward BFS ensures nodes are reachable from the start; backward BFS ensures nodes can reach the end.  
4. Iterative convergence: repeat reachability filtering until stable.

Effect: typically reduces O(nm) nodes down to hundreds or thousands, significantly accelerating later stages.

#### Stage 1: Single-stage Dynamic Programming

Perform DP over the stable node set to maximize total similarity (the sum of edge weights); average similarity is reported after path extraction:

- Objective: maximize (sum of edge weights) / (number of operations).  
- Transition: for each node (i, j), attempt transfers from three possible predecessors corresponding to 1:1, 2:1, and 1:2 operations, take the max cumulative score.  
- Path reconstruction: backtrack parent pointers from the terminal node to get the optimal node sequence.  
- Operation extraction: differences between adjacent nodes determine operation type and similarity score.

Guarantee: the DP finds the optimal path among all paths that satisfy the hard/soft constraints and reachability, from (0,0) to (n,m).

#### Stage 2: Final Structural Validation

Perform structural checks on the final alignment path; dynamic penalties are not applied during search:

- Purpose: detect pathological alignment patterns (consecutive different non-1:1 operations).  
- Rule: if two different non-1:1 operations occur within distance L (lookahead, default 5), mark as an exception.  
- Reporting: exceptions are recorded in repair logs for later analysis or manual review.

This design preserves fairness during search while identifying structural issues for subsequent handling.

### Post-processing Repairs

For DP-identified non-1:1 operations, attempt local repairs to convert them to 1:1 alignments:

**2:1 (source ahead) handling**:
- Try hard split of the source line at punctuation boundaries.  
- Try soft split based on semantic similarity (choose split that maximizes combined similarity).  
- If both fail, insert a synthetic placeholder line into the target.

**1:2 (target split) handling**:
- Merge the two target lines into one.

Application strategy:
- Process repair candidates in descending order of target index to avoid index drift.  
- Rebuild index mapping after each repair to ensure consistency.  
- Synthetic inserted lines do not participate in global similarity calculation to avoid semantic pollution.

Automatic retry on index-shift anomalies:

- After repairs complete, the pipeline analyzes the repaired output for run-length
    index-shift anomalies (groups of consecutive lines where the best-matching target
    neighbor is at i-1 or i+1). If such anomalies are detected (codes `TARGET_BEFORE`,
    `TARGET_AFTER` or mixed runs), the system will automatically retry the entire
    alignment+repair pipeline with progressively more permissive `node_relative_threshold`
    values to allow a wider search space and potentially fix the misalignment.
- Relaxation schedule: the pipeline attempts up to three additional runs with
    cumulative relaxations of 0.05, 0.15 and 0.30 from the original threshold
    (i.e. new_threshold = orig_threshold - cumulative). Lower threshold means
    the node-level filter is more permissive.
- The pipeline only returns the final (successful or last) repair logs; intermediate
    attempts are not written to persistent logs to avoid confusion.

## Configuration Parameters

### Major parameters

| Parameter | Default | Type | Description |
|---|---:|---|---|
| `model_name` | `"Alibaba-NLP/GTE-multilingual-base"` | str | Sentence embedding model name |
| `device` | `"cpu"` | str | Compute device (`"cpu"` or `"cuda"`) |
| `node_relative_threshold` | `0.8` | float | Node-level soft constraint relative threshold θ |
| `consecutive_non_1to1_lookahead` | `5` | int | Lookahead window L for structural validation |
| `soft_split_penalty` | `0.05` | float | Penalty for soft splits |
| `insert_fallback_score` | `0.6` | float | Fixed similarity score for inserted placeholders |
| `delete_penalty` | `0.05` | float | Penalty for delete operations (reserved) |

### Parameter tuning

#### By language pair

1. High-similarity pairs (e.g., English–French, Chinese–Japanese)  
   - Recommend increasing `node_relative_threshold` to 0.85–0.95.  
   - Result: stricter search space and more 1:1 alignments.

2. Low-similarity pairs (e.g., English–Chinese, English–Arabic)  
   - Recommend lowering `node_relative_threshold` to 0.5–0.65.  
   - Result: more permissive search space to accommodate structural differences.

3. Pairs with large structural differences  
   - Recommend increasing `consecutive_non_1to1_lookahead` to 7–10.  
   - Result: stricter detection of pathological patterns.

#### Performance vs. quality trade-offs

- High-quality mode: use defaults (validated across many language pairs and text types).  
- Fast mode: increase `node_relative_threshold` or decrease `consecutive_non_1to1_lookahead`.  
- Special texts (poetry, code): tune `node_relative_threshold` to better fit structure.

## API Reference

### TextAligner class (summary)

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
        """Run repair pipeline and return a result dict"""
    
    def save_results(
        self,
        result: Dict[str, Any],
        output_dir: Optional[str] = None,
        logs_file: Optional[str] = None,
        repaired_file: Optional[str] = None,
        include_texts: bool = True
    )
        """Save repair results to an output directory"""
    
    def print_report(
        self,
        result: Dict[str, Any],
        output_dir: Optional[str] = None,
        show_sample_logs: int = 3
    )
        """Print a repair report"""
```

### Returned result structure

`repair()` returns a dictionary containing:

```
{
    "repaired_lines": [str],  # Repaired target text lines
    "repair_logs": [RepairLog],  # Detailed repair logs
    "line_number_mapping": LineNumberMapping,  # Original/repaired line mapping
    "stats": {  # Statistics
        "total_repairs": int,
        "source_content_lines": int,
        "target_content_lines_before": int,
        "target_content_lines_after": int,
        "line_difference": int,
        "similarity_improvement": float,
        "dp_time_seconds": float,
        "postprocess_time_seconds": float,
        "total_time_seconds": float,
        "dp_operations": {  # DP operation counts
            "NO_SHIFT": int,
            "SOURCE_AHEAD": int,
            "TARGET_SPLIT": int
        },
        "similarity_statistics": {  # Similarity stats
            "min": float,
            "max": float,
            "mean": float,
            "median": float,
            "std": float,
            "low_quality_count": int
        }
    },
    "state": {
        "exceptions": [str]  # Detected exceptions list
    }
}
```

## Performance

### Time complexity

- Stage 0 (constraint computation): O(nm)  
- Stage 1 (DP): O(nm)  
- Stage 2 (validation): O(|π|) = O(max(n, m))  
- Post-processing: O(#repairs · log #repairs)  
- Total: O(nm + #repairs · log #repairs) ≈ O(nm)

### Space complexity

- DP table and caches: O(nm)  
- Repair logs and results: O(n + m)  
- Total: O(nm)

### Empirical performance

Typical scenario (source 1000 lines, target 1000 lines):

- Single-stage DP: typically < 5s (CPU)  
- Post-processing repairs: typically < 2s  
- Total: < 10s (model loading time dependent)

## Quality Metrics

### Output quality evaluation

After repair, review these metrics:

- 1:1 alignment ratio: percentage of operations that are 1:1  
- Average similarity: average similarity score of the alignment path (0–1)  
- Repair rate: repairs as a proportion of source lines  
- Exception count: number of structural exceptions detected  
- Low-quality alignments: alignments with similarity < 0.6

### Guidance

| Metric | Excellent | Good | Needs improvement |
|---|---:|---:|---:|
| 1:1 ratio | > 95% | 80–95% | < 80% |
| Avg similarity | > 0.85 | 0.70–0.85 | < 0.70 |
| Repair rate | < 2% | 2–5% | > 5% |
| Exceptions | 0 | 1–2 | > 2 |

## Troubleshooting

### Common issues

#### 1. Excessive memory use  
Symptoms: high memory consumption or OOM errors  
- Reduce text length or process in chunks.  
- Use a smaller embedding model.  
- Increase `node_relative_threshold` to reduce node count.

#### 2. Poor alignment quality  
Symptoms: low average similarity; many low-quality alignments  
- Check text preprocessing (sentence splitting correctness).  
- Adjust `node_relative_threshold` (lower allows more exploration).  
- Try a different embedding model.  
- Verify source and target texts are truly parallel.

#### 3. Slow performance  
Symptoms: DP stage is slow  
- Use a GPU if available (`device="cuda"`).  
- Increase `node_relative_threshold` to reduce node count.  
- Consider using a smaller/faster embedding model.

#### 4. Excessive repair failures or exceptions  
Symptoms: post-processing errors or too many exceptions reported  
- Check if source/target line count difference is large (> 50%).  
- Adjust `consecutive_non_1to1_lookahead` to change exception sensitivity.  
- Verify whether texts are actually parallel.

### Error messages explained

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Source file not found` | Source file path does not exist | Check file path |
| `No stable nodes found` | Constraints too strict; no feasible nodes | Lower `node_relative_threshold` |
| `LineDifferenceException` | Repaired source/target line counts mismatch | Check repair logic or input texts |
| `RepairRateException` | Repair rate exceeds 5% | Inspect text quality or adjust parameters |
| `LowSimilarityException` | Multiple low-similarity alignments | Check language pair or model choice |

## Advanced Usage

### Custom embedding model

```python
from sentence_transformers import SentenceTransformer

aligner = TextAligner(
    source_path="source.txt",
    target_path="target.txt",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
```

### Custom parameters

```python
aligner = TextAligner(
    source_path="source.txt",
    target_path="target.txt",
    node_relative_threshold=0.8,
    consecutive_non_1to1_lookahead=7,
    device="cuda"
)
```

### Batch processing

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

### Inspect detailed repair logs

```python
result = aligner.repair()

# Iterate repairs
for repair_log in result["repair_logs"]:
    print(f"Repair type: {repair_log.repair_type}")
    print(f"Similarity improvement: {repair_log.similarity_after - repair_log.similarity_before:.3f}")
    print(f"Source range: {repair_log.src_start}-{repair_log.src_end}")
    print(f"Target range: {repair_log.tgt_start}-{repair_log.tgt_end}")
    print()

# Stats
stats = result["stats"]
print(f"NO_SHIFT count: {stats['dp_operations']['NO_SHIFT']}")
print(f"Total repairs: {stats['total_repairs']}")
print(f"Average similarity: {stats['similarity_statistics']['mean']:.3f}")
print(f"Exception count: {len(result['state']['exceptions'])}")
```

## Development & Contribution

### Project layout

```
bilingual_aligner/
├── __init__.py
├── api.py                    # High-level API
├── cli.py                    # Command-line interface
├── corpus.py                 # Corpus and similarity computation
├── position.py               # Line-number mapping utilities
├── utils.py                  # Utility functions
├── core/
│   ├── processor.py          # Embedding & encoding
│   ├── splitter.py           # Sentence splitting
│   ├── punctuation.py        # Punctuation handling
│   └── repairer.py           # Repair coordinator
├── alignment/
│   ├── base.py               # Base classes
│   └── enum_aligner.py       # Single-stage DP algorithm
├── repair/
│   ├── models.py             # Data models
│   ├── executor.py           # Repair executor
│   └── coordinator.py        # Repair coordinator
└── analyzer/                 # Analysis tools
    ├── base.py
    ├── similarity.py
    ├── encoding.py
    ├── punctuation.py
    └── comparison.py
```

### Run tests

```bash
pip install bilingual-aligner[dev]
pytest tests/ -v
```

### Code style

```bash
black bilingual_aligner/
flake8 bilingual_aligner/
```

## Citation

If you use this library in research, please cite:

```bibtex
@software{bilingual_aligner,
  title = {Bilingual Aligner: optimized single-stage dynamic programming for bilingual alignment},
  author = {Elysia},
  year = {2024},
  url = {https://github.com/LoveElysia1314/bilingual_aligner}
}
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Support

- Issues: [GitHub Issues](https://github.com/LoveElysia1314/bilingual_aligner/issues)  
- Email: dr.zqr@outlook.com  
- Detailed documentation: see ALGORITHM.md

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) — sentence embedding library  
- [Alibaba NLP](https://huggingface.co/Alibaba-NLP) — pretrained multilingual models  
- Thanks to all contributors and users for feedback and support.