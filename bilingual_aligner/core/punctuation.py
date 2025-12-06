import re
from typing import Optional


# =============================
# 统一标点处理器
# =============================


class PunctuationHandler:
    """Unified handling of all punctuation-related logic to eliminate code duplication"""

    # Define all punctuation patterns centrally
    ALL_PUNCT_PATTERN = re.compile(r"[^\w\s\u4e00-\u9fff]")

    @staticmethod
    def is_between_ascii_letters(text: str, pos: int) -> bool:
        """
        Check if punctuation is surrounded by ASCII letters.

        Used for:
        - count_punctuation_line() - skip counting these punctuation marks
        - splitting logic - avoid splitting at these punctuation marks

        Args:
            text: text string
            pos: punctuation position

        Returns:
            True if punctuation is surrounded by ASCII letters
        """
        if 0 < pos < len(text) - 1:
            left = text[pos - 1]
            right = text[pos + 1]
            # Only treat ASCII letters as abbreviation context. Using
            # str.isalpha() here would also return True for CJK characters
            # which would incorrectly cause Chinese punctuation to be
            # skipped. Use isascii()+isalpha() to restrict to ASCII letters.
            try:
                left_ok = left.isascii() and (left.isalpha() or left.isdigit())
                right_ok = right.isascii() and (right.isalpha() or right.isdigit())
                return left_ok and right_ok
            except Exception:
                # Fallback conservative behavior: do not treat as ASCII-abbrev
                return False
        return False

    @staticmethod
    def is_decimal_point(text: str, pos: int) -> bool:
        """
        Check if position is a decimal point (e.g., . in 3.14).

        Args:
            text: text string
            pos: position

        Returns:
            True if it is a decimal point
        """
        if text[pos] == "." and 0 < pos < len(text) - 1:
            return text[pos - 1].isdigit() and text[pos + 1].isdigit()
        return False

    @staticmethod
    def should_skip_for_splitting(text: str, pos: int) -> bool:
        """
        Check if punctuation should be skipped during splitting.

        Rules:
        1. Skip punctuation surrounded by ASCII letters (English abbreviations)
        2. Skip decimal points

        Args:
            text: text string
            pos: punctuation position

        Returns:
            True if this split point should be skipped
        """
        return PunctuationHandler.is_between_ascii_letters(
            text, pos
        ) or PunctuationHandler.is_decimal_point(text, pos)

    @staticmethod
    def count_punctuation_line(line: str) -> int:
        """
        Count punctuation marks in a line of text.

        Rules:
        - Skip punctuation surrounded by ASCII letters (English abbreviations)
        - Count consecutive identical punctuation as one

        Args:
            line: input text line

        Returns:
            number of punctuation marks
        """
        puncts = []
        prev_end = -1
        prev_punct = None

        for m in PunctuationHandler.ALL_PUNCT_PATTERN.finditer(line):
            start, end = m.span()
            punct = m.group()

            # Skip English abbreviations (surrounded by ASCII letters)
            if PunctuationHandler.is_between_ascii_letters(line, start):
                continue

            # Count consecutive identical punctuation as one: merge only if adjacent in original text
            if start != prev_end or punct != prev_punct:
                puncts.append(punct)

            prev_end = end
            prev_punct = punct

        return len(puncts)


# =============================
# Punctuation similarity calculation (for alignment quality assessment)
# =============================
def calculate_punctuation_similarity(
    src_text: str, tgt_text: str, tolerance: float = 2.5, divisor: float = 10.0
) -> float:
    """
    Calculate punctuation similarity weight coefficient using quadratic penalty function.

    Formula:
    weight = max(0, (1 - (|Δp| - tolerance) / divisor)^2)

    Where:
    - |Δp| = |p_src - p_tgt| = absolute punctuation count difference
    - tolerance = threshold before penalty starts (default 2.5)
      When |Δp| ≤ tolerance, weight = 1 (no penalty)
    - divisor = scaling factor for penalty steepness (default 10)
      Larger divisor = gentler penalty; Smaller divisor = stricter penalty

    Advantages of quadratic formula:
    1. Complete fairness: identical punctuation differences yield identical weights regardless of sentence length
    2. Adaptive penalty: smooth quadratic curve with increasing penalty for larger differences
    3. Intuitive parameters: only two adjustable values (tolerance and divisor)
    4. Stable distribution: most weight values cluster around high confidence (>0.8)

    Example progression (with default tolerance=2.5, divisor=10):
    - |Δp| = 0,1,2: weight = 1.0 (no penalty, within tolerance)
    - |Δp| = 2.5: weight = 1.0 (exactly at tolerance boundary)
    - |Δp| = 3: weight = 0.7225 (penalty starts at Δp=3)
    - |Δp| = 4: weight = 0.5625
    - |Δp| = 5: weight = 0.4225
    - |Δp| = 7: weight = 0.1225
    - |Δp| = 12.5: weight = 0 (complete rejection)

    Args:
        src_text: source text
        tgt_text: target text
        tolerance: punctuation difference tolerance (default 2.5, no penalty if diff ≤ tolerance)
        divisor: scaling factor controlling penalty steepness (default 10)

    Returns:
        weight coefficient, range [0, 1]
    """
    # Calculate punctuation counts for both texts
    p_src = PunctuationHandler.count_punctuation_line(src_text)
    p_tgt = PunctuationHandler.count_punctuation_line(tgt_text)

    # Absolute difference in punctuation counts
    punct_diff = abs(p_src - p_tgt)

    # Effective difference: only penalize beyond tolerance threshold
    effective_diff = max(0, punct_diff - tolerance)

    # Quadratic penalty: (1 - normalized_diff)^2
    # The formula is: max(0, (1 - effective_diff / divisor)^2)
    base = 1.0 - effective_diff / divisor
    weight = max(0, base**2)

    return weight
