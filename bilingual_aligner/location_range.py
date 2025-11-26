"""Compatibility shim for `location_range`.

This module is retained to preserve the original import path while the
implementation has been consolidated into `bilingual_aligner.position`.
It re-exports the primary symbols and emits a DeprecationWarning on import.
"""

import warnings

from .position import LocationRange

warnings.warn(
    "bilingual_aligner.location_range is deprecated; use bilingual_aligner.position instead",
    DeprecationWarning,
)

__all__ = ["LocationRange"]
