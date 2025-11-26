"""Compatibility shim for `line_number_manager`.

The implementation has been moved to `bilingual_aligner.position`. This
module re-exports the primary symbols to preserve backward compatibility
and emits a DeprecationWarning on import.
"""

import warnings

from .position import LineNumberMapping, build_line_number_mapping

warnings.warn(
    "bilingual_aligner.line_number_manager is deprecated; use bilingual_aligner.position instead",
    DeprecationWarning,
)

__all__ = ["LineNumberMapping", "build_line_number_mapping"]
