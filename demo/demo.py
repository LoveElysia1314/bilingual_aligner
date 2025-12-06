#!/usr/bin/env python3
"""
Bilingual Aligner Demo Script

This script demonstrates basic usage of the bilingual_aligner API.

Usage:
    python demo.py

Requirements:
    - Install the package: pip install -e .
    - Run from the project root directory
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bilingual_aligner import TextAligner


def main():
    """Main demo function"""

    # Set demo logging level to DEBUG
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Disable propagation to avoid duplicate output
    bilingual_logger = logging.getLogger("bilingual_aligner")
    bilingual_logger.propagate = False

    # Define file paths
    demo_dir = Path(__file__).parent
    source_file = demo_dir / "sample_zh.md"
    target_file = demo_dir / "sample_en.md"
    output_dir = demo_dir / "output"

    print("Bilingual Aligner Demo")
    print("=" * 50)
    print(f"Source file: {source_file}")
    print(f"Target file: {target_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Initialize the aligner with custom configuration
    config = {"node_relative_threshold": 0.8}  # Test different threshold values

    # Direct API call
    aligner = TextAligner(str(source_file), str(target_file), **config)
    result = aligner.repair()

    # Save results
    aligner.save_results(result, str(output_dir))

    # Print summary
    aligner.print_report(result, str(output_dir))

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    sys.exit(main())
