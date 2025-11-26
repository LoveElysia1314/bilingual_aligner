"""
Utility functions for the bilingual aligner.
"""

import hashlib


def sha256(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def build_config_from_args(args):
    """Build core TextAligner config and optional TextProcessor from argparse Namespace.

    Returns: (config_dict, text_processor_or_none)
    """
    config = {
        "model_name": getattr(args, "model", None),
        "k": getattr(args, "k", None),
        "soft_split_penalty": getattr(args, "soft_split_penalty", None),
        "insert_fallback_score": getattr(args, "insert_fallback_score", None),
    }

    # Remove None values to avoid overriding defaults
    config = {k: v for k, v in config.items() if v is not None}

    # Lazy creation of TextProcessor is handled by get_text_processor; do not
    # create processor here to avoid unnecessary model loads. Caller may choose
    # to call get_text_processor when they actually need a processor instance.
    return config, None
