# Bilingual Aligner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python >=3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/bilingual-aligner.svg)](https://pypi.org/project/bilingual-aligner/)

A high-performance bilingual text alignment and repair library, implementing **optimized single-stage dynamic programming algorithm** with pruning-based path enumeration.

## Features

- **Optimized Single-Stage DP Algorithm**: Replaces two-stage DP with single-stage DP and pruning-based enumeration
- **Pruning Theorem**: Early stopping when `max(adjusted score) > max(unadjusted score_remaining)` to ensure global optimality
 - **Final Structural Validation**: Final check for consecutive different non-1:1 operations (no dynamic penalties applied during search)
- **Post-processing Repair**: Local repair for non-1:1 alignments (split, merge, insert)
- **Embedding-based Similarity**: Uses sentence-transformers with weighted punctuation
- **Multi-language Support**: Default model: Alibaba-NLP/GTE-multilingual-base
- **High Performance**: O(nm) time complexity, actual speedup from pruning

## Installation

```bash
pip install bilingual-aligner
```

Install development dependencies:
```bash
pip install bilingual-aligner[dev]
```

## Quick Start

```python
from bilingual_aligner import TextAligner

# Initialize aligner
aligner = TextAligner(
    model_name="Alibaba-NLP/GTE-multilingual-base",
    device="cpu"  # or "cuda" for GPU
)

# Load bilingual text
source_text = "Hello world.\nThis is a test."
target_text = "你好世界。\n这是一个测试。"

# Align and repair
aligner.align(source_text, target_text)
aligner.repair()

# Save results
aligner.save_results("output/")
aligner.print_report()
```

## Algorithm Overview

### Optimized Single-Stage DP with Pruning-based Enumeration

The algorithm simplifies the DP approach:

1. **Stage 0: Constraint Node Calculation**
   - Hard constraints: `ceil(i/2) ≤ j ≤ 2i` and `ceil((n-i)/2) ≤ m-j ≤ 2(n-i)`
    - Soft constraints: handled via node-level filtering and operation-level policies
   - Stability filtering: Forward/backward BFS to find stable nodes

2. **Stage 1: Single-Stage DP with Pruning-based Enumeration**
   - Standard DP computes all reachable paths (no dynamic penalty)
   - Paths enumerated and sorted by unpenalized score in descending order
    - Paths enumerated and scored by average similarity; final structural validation is applied after selection
   - Early stopping via pruning theorem, ensuring global optimality

### Final Structural Validation

The dynamic penalty mechanism has been removed from the path scoring stage. The algorithm now:

- Enumerates paths and ranks them by average similarity (primary objective).
- Applies a final structural validation on the selected path to detect consecutive different non-1:1 operations within a small lookahead window; if found, the aligner reports this as an exception for downstream handling.

This ensures fair treatment of all operation types during search while still flagging structural inconsistencies in the final result.

### Pruning Theorem

**Theorem**: If there exists k₀ such that `max_adjusted_score(k₀) > max_unadjusted_score`, then for all k ≥ k₀:
```
max_{i>k₀} adjusted_score(π_i) < max_adjusted_score(k₀)
```

This allows early stopping while guaranteeing global optimality.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | "Alibaba-NLP/GTE-multilingual-base" | Sentence embedding model name to use |
| `k` | 0.6 | Tolerance parameter for punctuation weight calculation |
| `soft_split_penalty` | 0.05 | Penalty for soft splits between sentences |
| `insert_fallback_score` | 0.6 | Fixed score for insert fallback option |
| `delete_penalty` | 0.05 | Penalty for delete operations (future use) |
| `consecutive_non_1to1_penalty_max` | (deprecated) | Deprecated: dynamic penalty removed; unused |
| `consecutive_non_1to1_lookahead` | 5 | Lookahead steps for final structural validation |
| `MIN_QUALITY_THRESHOLD` | (deprecated) | Legacy parameter; no longer used in current single-stage flow |

## API Reference

### TextAligner Class

```python
class TextAligner:
    def __init__(self, source_path, target_path, model_name="Alibaba-NLP/GTE-multilingual-base", **config):
        """Initialize aligner
        
        Args:
            source_path (str): Path to source file
            target_path (str): Path to target file
            model_name (str): Embedding model name
            **config: Additional configuration parameters
        """
    
    def align(self, source_text, target_text):
        """Align source and target text (deprecated, use repair())"""
    
    def repair(self):
        """Execute the repair process and return results"""
    
    def save_results(self, output_dir, **kwargs):
        """Save repair results"""
    
    def print_report(self):
        """Print repair report"""
```

#### API Configuration Parameters

The following configuration parameters can be passed via `**config`:

- `k`: Tolerance parameter for punctuation weight calculation (default: 0.6)
- `soft_split_penalty`: Penalty for soft splits between sentences (default: 0.05)
- `insert_fallback_score`: Fixed score for insert fallback option (default: 0.6)
- `delete_penalty`: Penalty for delete operations (default: 0.05)
- `consecutive_non_1to1_penalty_max`: Deprecated (dynamic penalty removed)
- `consecutive_non_1to1_lookahead`: Lookahead steps for final structural validation (default: 5)

### Command Line Interface

```bash
# Basic alignment
bilingual-aligner align source.txt target.txt --output output/

# Use custom model
bilingual-aligner align source.txt target.txt --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Repair-only mode
bilingual-aligner repair alignment.json --output repaired/
```

#### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source`, `-s` | - | Path to the source language file (required) |
| `--target`, `-t` | - | Path to the target language file (required) |
| `--output`, `-o` | - | Output directory for results (required) |
| `--model` | None | Sentence embedding model name to use |
| `--k` | 0.8 | Tolerance parameter for punctuation weight calculation |
| `--soft-split-penalty` | 0.05 | Penalty for soft splits between sentences |
| `--insert-fallback-score` | 0.6 | Fixed score for insert fallback option |
| `--logs-file` | repair_logs.json | Filename for repair logs |
| `--repaired-file` | <target_basename>_repaired.txt | Filename for repaired target output |
| `--no-texts` | False | Do not include full source/target texts in logs |
| `--verbose`, `-v` | False | Enable verbose output |

## Performance

### Time Complexity
- **Stage 0**: O(nm) - Constraint calculation
- **Stage 1**: O(nm + P·L) - DP + path enumeration (P: number of paths, L: path length)
- **With pruning**: Typically O(nm), early stopping

### Memory Usage
- O(nm) for DP table
- O(P·L) for path storage

## Advanced Usage

### Custom Embedding Model

```python
from sentence_transformers import SentenceTransformer

# Use custom model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
aligner = TextAligner(model_name=model, device="cuda")
```

### Batch Processing

```python
# Process multiple files
for src_file, tgt_file in file_pairs:
    with open(src_file, 'r', encoding='utf-8') as f:
        source_text = f.read()
    with open(tgt_file, 'r', encoding='utf-8') as f:
        target_text = f.read()
    
    aligner.align(source_text, target_text)
    aligner.repair()
    aligner.save_results(f"output/{src_file.stem}/")
```

### Quality Metrics

```python
# Access alignment statistics
stats = aligner.get_statistics()
print(f"1:1 ratio: {stats['one_to_one_ratio']:.2%}")
print(f"Average similarity: {stats['avg_similarity']:.3f}")
print(f"Non-1:1 operations: {stats['non_one_to_one_count']}")
```

## Troubleshooting

### Common Issues

1. **High memory usage**
   - Reduce batch size for embedding computation
   - Use smaller embedding models
   - Enable pruning for early stopping

2. **Poor alignment quality**
    - Check text preprocessing (sentence splitting, normalization)
    - Try different embedding models
    - Adjust node-level filtering (`node_relative_threshold`) or other alignment thresholds

3. **Slow performance**
   - Use GPU if available (`device="cuda"`)
   - Enable pruning (enabled by default)
   - Reduce text length or use chunked processing

### Error Messages

- `"Stable nodes not found"`: Check text length and similarity scores
- `"Embedding model not found"`: Verify model name or install sentence-transformers
- `"Index out of range"`: Check text preprocessing and line counts

## Development

### Project Structure

```
bilingual_aligner/
├── __init__.py              # Package exports
├── api.py                   # TextAligner class (high-level API)
├── cli.py                   # Command line interface
├── corpus.py                # Corpus and similarity calculation
├── position.py              # LocationRange and line number mapping utilities
├── repair_applier.py        # Post-processing repair
├── utils.py                 # Utility functions
├── aligners/                # DP alignment algorithm implementations
│   ├── __init__.py
│   ├── enum_pruning_aligner.py  # v3.0 algorithm implementation
│   └── dp_aligner_two_stage.py  # Legacy v2.1 implementation
└── core/                    # Core text processing modules
    ├── __init__.py
    ├── processor.py         # TextProcessor class
    ├── punctuation.py       # Punctuation handling
    ├── repairer.py          # Main repair orchestrator
    └── splitter.py          # Universal sentence splitter
```

### Running Tests

```bash
# Install development dependencies
pip install bilingual-aligner[dev]

# Run tests
pytest tests/
```

### Code Style

```bash
# Format code
black bilingual_aligner/

# Code checking
flake8 bilingual_aligner/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{bilingual_aligner,
  title = {Bilingual Aligner: Optimized Single-Stage DP Algorithm for Text Alignment},
  author = {Elysia},
  year = {2024},
  url = {https://github.com/LoveElysia1314/bilingual_aligner}
}
```

## Acknowledgments

- Sentence-transformers library for embedding models
- Model Context Protocol (MCP) community
- Contributors and users of the bilingual-aligner project

## Support

- **Issues**: [GitHub Issues](https://github.com/LoveElysia1314/bilingual_aligner/issues)
- **Email**: dr.zqr@outlook.com
- **Documentation**: See [ALGORITHM.md](ALGORITHM.md) for detailed algorithm description