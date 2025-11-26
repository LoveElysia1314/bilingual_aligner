import argparse
import os
import logging
from .api import TextAligner, calculate_similarity
from .utils import build_config_from_args


def main():
    parser = argparse.ArgumentParser(
        description="Bilingual Text Alignment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bilingual-aligner --source source.txt --target target.txt --output results/
  bilingual-aligner -s cn.txt -t en.txt -o output/ --max-repairs 50
        """,
    )

    parser.add_argument(
        "-s", "--source", required=False, help="Path to the source language file"
    )
    parser.add_argument(
        "-t", "--target", required=False, help="Path to the target language file"
    )
    parser.add_argument(
        "-o", "--output", required=False, help="Output directory for results"
    )

    # Similarity test mode (lightweight utility)
    parser.add_argument(
        "--similarity",
        action="store_true",
        help="Run a one-off similarity check between two text snippets (uses embedding model if available)",
    )
    parser.add_argument("--text1", type=str, help="First text for similarity test")
    parser.add_argument("--text2", type=str, help="Second text for similarity test")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional sentence-transformers model name to use for similarity",
    )
    parser.add_argument(
        "--soft-split-penalty",
        type=float,
        default=0.05,
        help="Penalty for soft splits between sentences (default: 0.05)",
    )
    parser.add_argument(
        "--insert-fallback-score",
        type=float,
        default=0.6,
        help="Score for insert fallback option (default: 0.6)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (show debug information)",
    )

    parser.add_argument(
        "--logs-file",
        type=str,
        help="Filename for repair logs. If a simple filename is provided, it will be written into --output. If a path with directories is provided, it will be treated as relative to --output. Absolute paths are used as-is.",
    )

    parser.add_argument(
        "--repaired-file",
        type=str,
        help="Filename for repaired target output. If a simple filename is provided, it will be written into --output. If a path with directories is provided, it will be treated as relative to --output. Absolute paths are used as-is.",
    )

    parser.add_argument(
        "--no-texts",
        action="store_true",
        help="Do not include full source/target texts in repair logs",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # If user requested similarity mode, validate text args
    if args.similarity:
        if not args.text1 or not args.text2:
            logging.error(
                "Error: --text1 and --text2 are required when using --similarity"
            )
            return 1
        # Run similarity check via API helper (includes fallback)
        try:
            sim = calculate_similarity(args.text1, args.text2, model_name=args.model)
            print(f"Similarity (cosine) = {sim:.6f}")
        except Exception as e:
            logging.warning(f"Similarity check failed: {e}")
            t1 = args.text1.strip()
            t2 = args.text2.strip()
            fallback = 1.0 if t1 == t2 else 0.0
            print(f"Fallback similarity = {fallback:.6f}")
        return 0

    # Validate input files for repair mode
    if not args.source or not os.path.exists(args.source):
        logging.error(
            f"Error: Source file '{args.source}' does not exist or not provided"
        )
        return 1

    if not args.target or not os.path.exists(args.target):
        logging.error(
            f"Error: Target file '{args.target}' does not exist or not provided"
        )
        return 1

    if not args.output:
        logging.error("Error: --output is required for repair mode")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize aligner with config (use helper to centralize mapping)
    config, _ = build_config_from_args(args)

    try:
        aligner = TextAligner(args.source, args.target, **config)
        result = aligner.repair()

        # Ensure output dir exists
        os.makedirs(args.output, exist_ok=True)

        # Note: Directory creation for logs_file and repaired_file is now handled
        # by save_results() to ensure consistent behavior

        # Persist results and print a reusable demo-style report
        saved_result = aligner.save_results(
            result,
            args.output,
            logs_file=args.logs_file,
            repaired_file=args.repaired_file,
            include_texts=(not args.no_texts),
        )

        # Print report and final IO locations
        aligner.print_report(result, args.output, show_sample_logs=20)
        if saved_result and isinstance(saved_result, dict):
            io_info = saved_result.get("io", {})
            if io_info:
                logging.info(f"Repaired file written: {io_info.get('repaired_path')}")
                logging.info(f"Repair logs written: {io_info.get('logs_path')}")

        logging.info("Alignment repair completed successfully!")
        return 0
    except Exception as e:
        logging.error(f"Error during alignment: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
