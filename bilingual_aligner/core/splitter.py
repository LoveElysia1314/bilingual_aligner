import re

from typing import List, Tuple
from .punctuation import PunctuationHandler


class UniversalSplitter:
    """Language-independent minimal sentence splitter, supports mixed Chinese-English text, avoids splitting inside quotes/brackets"""

    # Unified sentence endings
    SENTENCE_END_PATTERN = r"[.!?。！？]"
    # Allow splitting on commas and colons (both half-width and full-width)
    SOFT_SPLIT_PATTERN = r"[,:，：]"

    # Quote/bracket pairs (open → close)
    PAIRS = {
        '"': '"',
        "'": "'",
        "“": "”",
        "「": "」",
        "『": "』",
        "(": ")",
        "（": "）",
        "[": "]",
        "【": "】",
        "{": "}",
        "｛": "｝",
    }
    # All opening and closing symbols
    OPENERS = set(PAIRS.keys())
    CLOSERS = set(PAIRS.values())

    @classmethod
    def _is_part_of_ellipsis(cls, text: str, pos: int) -> bool:
        """Check if the '.' at position pos is part of an ellipsis (3+ consecutive dots)"""
        if pos >= len(text) or text[pos] != ".":
            return False

        # Count consecutive dots around pos
        start = pos
        while start > 0 and text[start - 1] == ".":
            start -= 1
        end = pos
        while end < len(text) - 1 and text[end + 1] == ".":
            end += 1

        dot_count = end - start + 1
        return dot_count >= 3

    @classmethod
    def _build_quote_context(cls, text: str) -> List[bool]:
        """
        Returns a boolean list in_quoted, where in_quoted[i] == True means position i is inside some quote/bracket pair.
        Note: Only handles properly paired symbols, ignores unpaired ones (like single ").

        For symmetric quotes/brackets (where opener == closer, like "), we use a toggling strategy:
        the first occurrence is an opener, the second is a closer, etc.
        """
        n = len(text)
        in_quoted = [False] * n
        stack = []  # Store (opener_char, start_index)

        i = 0
        while i < n:
            char = text[i]
            # 如果该字符是被 ASCII 字母/数字包围的半角标点（例如 Alya's 中的 '），
            # 则应视为单词内部，不当作引号/括号 opener/closer 处理。
            # 使用 PunctuationHandler.is_between_ascii_letters 保持与标点统计/跳过逻辑一致。
            try:
                if PunctuationHandler.is_between_ascii_letters(text, i):
                    i += 1
                    continue
            except Exception:
                # 保守处理：若检查失败，继续原有逻辑
                pass
            # Check if this character is a symmetric quote/bracket (both opener and closer)
            if (
                char in cls.OPENERS
                and char in cls.CLOSERS
                and cls.PAIRS.get(char) == char
            ):
                # Symmetric character: toggle behavior based on stack
                if stack and stack[-1][0] == char:
                    # Top of stack is the same character, treat as closer
                    _, start = stack.pop()
                    # Mark [start+1, i-1] as in_quoted
                    for j in range(start + 1, i):
                        if j < n:
                            in_quoted[j] = True
                else:
                    # Stack is empty or top is different, treat as opener
                    stack.append((char, i))
            elif char in cls.OPENERS:
                stack.append((char, i))
            elif char in cls.CLOSERS:
                if stack:
                    last_opener, start = stack[-1]
                    expected_closer = cls.PAIRS[last_opener]
                    if char == expected_closer:
                        # Pair matches, mark [start+1, i-1] as in_quoted
                        for j in range(start + 1, i):
                            if j < n:
                                in_quoted[j] = True
                        stack.pop()
                    else:
                        # Type mismatch (like 「 matching )), don't pop stack, treat as invalid
                        pass
                # else: No matching opener, ignore
            i += 1

        # Unclosed opening symbols? Ignore (don't mark as in_quoted)
        return in_quoted

    @classmethod
    def _find_quote_pairs(cls, text: str) -> List[Tuple[int, int]]:
        """
        Returns a list of properly matched quote/bracket pairs as (start_index, end_index).
        Returns only properly paired symbols, ignores unpaired ones.

        Note: For symmetric quotes/brackets (where opener == closer, like "), we use a
        toggling strategy: the first occurrence is an opener, the second is a closer, etc.
        """
        n = len(text)
        pairs = []
        stack = []  # Store (opener_char, start_index)

        i = 0
        while i < n:
            char = text[i]
            # 同样：跳过被 ASCII 字母/数字包围的半角符号，不将其视为引号/括号的开/闭符
            try:
                if PunctuationHandler.is_between_ascii_letters(text, i):
                    i += 1
                    continue
            except Exception:
                pass
            # Check if this character is a symmetric quote/bracket (both opener and closer)
            if (
                char in cls.OPENERS
                and char in cls.CLOSERS
                and cls.PAIRS.get(char) == char
            ):
                # Symmetric character: toggle behavior based on stack
                if stack and stack[-1][0] == char:
                    # Top of stack is the same character, treat as closer
                    _, start = stack.pop()
                    pairs.append((start, i))
                else:
                    # Stack is empty or top is different, treat as opener
                    stack.append((char, i))
            elif char in cls.OPENERS:
                stack.append((char, i))
            elif char in cls.CLOSERS:
                if stack:
                    last_opener, start = stack[-1]
                    expected_closer = cls.PAIRS[last_opener]
                    if char == expected_closer:
                        # Pair matches
                        pairs.append((start, i))
                        stack.pop()
                    else:
                        # Type mismatch, don't pop stack, treat as invalid
                        pass
                # else: No matching opener, ignore
            i += 1

        return pairs

    @classmethod
    def _get_safe_split_points_for_splitting(cls, text: str) -> List[int]:
        """
        For splitting: skip sentence-ending punctuation inside quotes/brackets, avoid splitting at abbreviations.
        Ellipsis regions are now allowed as split points.
        """
        if not text:
            return []
        in_quoted = cls._build_quote_context(text)
        points = []

        # First handle strong sentence-ending punctuation
        for match in re.finditer(cls.SENTENCE_END_PATTERN, text):
            i = match.start()
            char = match.group()
            pos = match.end()

            # Ellipsis regions are now allowed - do not skip them

            # Skip decimal points
            if PunctuationHandler.is_decimal_point(text, i):
                continue

            # 【Critical addition】: If sentence-ending punctuation itself is inside quotes/brackets, cannot be a split point!
            if in_quoted[i]:
                continue

            # Skip punctuation surrounded by letters (consistent with punctuation counting logic)
            if PunctuationHandler.should_skip_for_splitting(text, i):
                continue

            # Skip if part of ellipsis (three or more consecutive dots)
            if cls._is_part_of_ellipsis(text, i):
                continue

            # Extend with closing symbols (like ？」)
            while pos < len(text) and text[pos] in cls.CLOSERS:
                pos += 1

            if 0 < pos < len(text):
                points.append(pos)

        # Then consider softer splitters: commas and colons — only if they are not inside quotes/brackets
        for match in re.finditer(cls.SOFT_SPLIT_PATTERN, text):
            i = match.start()
            char = match.group()
            pos = match.end()

            # 如果该软分隔符被 ASCII 字母包围（例如单词内部的撇号/连字符），跳过
            if PunctuationHandler.is_between_ascii_letters(text, i):
                continue

            # Skip if inside any quoted/bracketed context
            if in_quoted[i]:
                continue

            # Avoid splitting small constructs like single-character phrases: require some context
            # ensure there is a non-space character before and after
            if i - 1 < 0 or i + 1 >= len(text):
                continue
            if text[i - 1].isspace() or text[i + 1].isspace():
                # allow if there is content but not if separated by spaces only
                pass

            # Do not split when comma/colon are part of numeric formatting (e.g., 1,000)
            if char in {",", "，"}:
                if i > 0 and i + 1 < len(text):
                    if text[i - 1].isdigit() and text[i + 1].isdigit():
                        continue

            # Advance past closing closers (like ）」）
            while pos < len(text) and text[pos] in cls.CLOSERS:
                pos += 1

            if 0 < pos < len(text):
                points.append(pos)

        return points

    @classmethod
    def _get_safe_split_points_for_counting(cls, text: str) -> List[int]:
        """
        For counting: don't skip sentence-ending punctuation inside quotes.
        Ellipsis regions are now allowed as split points.
        """
        if not text:
            return []
        points = []
        for match in re.finditer(cls.SENTENCE_END_PATTERN, text):
            i = match.start()
            char = match.group()
            pos = match.end()

            # Ellipsis regions are now allowed - do not skip them

            if PunctuationHandler.is_decimal_point(text, i):
                continue

            # Skip punctuation surrounded by letters (consistent with punctuation counting logic)
            if PunctuationHandler.should_skip_for_splitting(text, i):
                continue

            # Skip if part of ellipsis (three or more consecutive dots)
            if cls._is_part_of_ellipsis(text, i):
                continue

            # Don't check in_quoted! Sentences inside quotes should also be counted
            while pos < len(text) and text[pos] in cls.CLOSERS:
                pos += 1

            if pos <= len(text):  # Allow at end
                points.append(pos)

        return points

    @classmethod
    def find_split_points(cls, text: str, num_chunks: int) -> List[int]:
        if not text.strip():
            return []
        points = cls._get_safe_split_points_for_splitting(text)
        return points

    @classmethod
    def _analyze_quote_pair_split_point(
        cls, text: str, close_pos: int, start_pos: int
    ) -> Tuple[bool, str]:
        """
        分析最外层引括号结束位置是否允许分割。

        Args:
            text: 文本
            close_pos: 引括号闭合的位置（闭合符号的位置）
            start_pos: 对应引括号开放的位置（开放符号的位置）

        Returns:
            (allow_split: bool, split_type: str)
            - allow_split: 是否允许分割
            - split_type: 'hard', 'soft', 或 'none'（禁止）

        逻辑：
        1. 检查引括号后是否紧跟标点符号 → 有则禁止分割（返回 'none'）
        2. 检查引括号前（左边，忽略空白）是否是硬分割标点 → 是则允许硬分割，否则允许软分割
        """
        # 检查闭合符号后是否紧跟标点符号
        pos_after_close = close_pos + 1
        if pos_after_close < len(text):
            char_after = text[pos_after_close]
            # 检查是否是标点符号或闭合符号
            if char_after in cls.CLOSERS or re.match(r"[.!?。！？]", char_after):
                return (False, "none")  # 禁止分割

        # 查找引括号前的非空白、非闭合符号内容
        pos_before_open = start_pos - 1
        while pos_before_open >= 0 and text[pos_before_open].isspace():
            pos_before_open -= 1

        # 检查前面是否是硬分割标点或其他内容
        if pos_before_open >= 0:
            char_before = text[pos_before_open]
            # 硬分割标点：句号、问号、感叹号（中英文）、省略号
            if re.match(r"[.!?。！？]", char_before):
                return (True, "hard")
            # 也可能是省略号的最后一个点
            if char_before == "." and pos_before_open >= 2:
                # 检查是否是省略号
                if cls._is_part_of_ellipsis(text, pos_before_open):
                    return (True, "hard")
            # 其他情况允许软分割
            return (True, "soft")

        # 文本开始处，允许软分割
        return (True, "soft")

    @classmethod
    def find_hard_split_points(cls, text: str) -> List[int]:
        """
        Find HARD split points (sentence-ending punctuation only, no soft splits on commas/colons).
        These preserve complete sentences and don't require re-encoding of new sentence structures.
        Ellipsis regions are now allowed as split points.

        Improved logic for quote/bracket handling:
        - Find outermost quote/bracket pair end positions
        - Check if split is allowed after the pair (based on following punctuation and preceding context)
        - Determine if split should be hard or soft based on what's before the opening bracket
        """
        if not text:
            return []
        in_quoted = cls._build_quote_context(text)
        quote_pairs = cls._find_quote_pairs(text)
        points = []

        # First, collect all hard split points from sentence-ending punctuation outside quotes
        for match in re.finditer(cls.SENTENCE_END_PATTERN, text):
            i = match.start()
            char = match.group()
            pos = match.end()

            # Skip decimal points
            if char == "." and i > 0 and i + 1 < len(text):
                if text[i - 1].isdigit() and text[i + 1].isdigit():
                    continue

            # Skip punctuation surrounded by letters
            if PunctuationHandler.should_skip_for_splitting(text, i):
                continue

            # Skip if part of ellipsis
            if cls._is_part_of_ellipsis(text, i):
                continue

            # Only add points outside quotes for hard splits
            if not in_quoted[i]:
                # Extend with closing symbols
                while pos < len(text) and text[pos] in cls.CLOSERS:
                    pos += 1
                if 0 < pos < len(text):
                    points.append(pos)

        # Second, analyze quote/bracket pairs for split opportunities
        for start, close in quote_pairs:
            # Analyze the split point after this quote/bracket pair
            allow_split, split_type = cls._analyze_quote_pair_split_point(
                text, close, start
            )

            if allow_split and split_type == "hard":
                # Allow hard split after the closing bracket
                pos = close + 1
                # Extend with additional closing symbols
                while pos < len(text) and text[pos] in cls.CLOSERS:
                    pos += 1
                if 0 < pos <= len(text) and pos not in points:
                    points.append(pos)

        return sorted(points)

    @classmethod
    def _find_unclosed_openers(cls, text: str) -> set:
        """
        Returns a set of positions where opening symbols are left unclosed.
        These positions should not be considered as split points.
        """
        n = len(text)
        stack = []  # Store (opener_char, start_index)
        unclosed_positions = set()

        i = 0
        while i < n:
            char = text[i]
            # 跳过像 Alya's 中被字母包围的半角标点，避免误判为未闭合 opener
            try:
                if PunctuationHandler.is_between_ascii_letters(text, i):
                    i += 1
                    continue
            except Exception:
                pass

            if char in cls.OPENERS:
                stack.append((char, i))
            elif char in cls.CLOSERS:
                if stack:
                    last_opener, start = stack[-1]
                    expected_closer = cls.PAIRS[last_opener]
                    if char == expected_closer:
                        # Pair matches, pop from stack
                        stack.pop()
                    else:
                        # Type mismatch, don't pop stack
                        pass
                # else: No matching opener, ignore
            i += 1

        # All remaining items in stack are unclosed openers
        for _, pos in stack:
            unclosed_positions.add(pos)

        return unclosed_positions

    @classmethod
    def find_soft_split_points(cls, text: str) -> List[int]:
        """
        Find SOFT split points (commas, colons, etc).
        These should only be used as fallback when hard splits don't improve alignment.

        Improved logic for quote/bracket handling:
        - Find outermost quote/bracket pair end positions
        - Check if split is allowed after the pair (based on following punctuation and preceding context)
        - Only add soft split if the preceding content is not a hard split point
        - Avoid duplicating split points from commas/colons that directly follow quotes
        """
        if not text:
            return []
        in_quoted = cls._build_quote_context(text)
        unclosed_openers = cls._find_unclosed_openers(text)
        quote_pairs = cls._find_quote_pairs(text)
        points = []

        # Collect soft splitters: commas and colons
        for match in re.finditer(cls.SOFT_SPLIT_PATTERN, text):
            i = match.start()
            char = match.group()
            pos = match.end()

            # Skip if inside any quoted/bracketed context
            # 如果该软分隔符是被 ASCII 字母包围的半角标点（例如 Alya's），则跳过
            if PunctuationHandler.is_between_ascii_letters(text, i):
                continue

            if in_quoted[i]:
                continue

            # Skip if this is an unclosed opening symbol
            if i in unclosed_openers:
                continue

            # Avoid splitting small constructs
            if i - 1 < 0 or i + 1 >= len(text):
                continue
            if text[i - 1].isspace() or text[i + 1].isspace():
                pass

            # Do not split when comma/colon are part of numeric formatting
            if char in {",", "，"}:
                if i > 0 and i + 1 < len(text):
                    if text[i - 1].isdigit() and text[i + 1].isdigit():
                        continue

            if 0 < pos < len(text):
                points.append(pos)

        # Also consider ellipsis (three or more consecutive dots) as soft split points
        i = 0
        while i < len(text):
            if text[i] == ".":
                # Count consecutive dots
                dot_start = i
                while i < len(text) and text[i] == ".":
                    i += 1
                dot_count = i - dot_start

                # If it's an ellipsis (3+ dots), add as soft split point
                if dot_count >= 3:
                    # Skip if inside quotes/brackets
                    if not in_quoted[dot_start]:
                        # Add split point after the ellipsis
                        if dot_start + dot_count < len(text):
                            points.append(dot_start + dot_count)
            else:
                i += 1

        # Analyze quote/bracket pairs for soft split opportunities
        # But ONLY if a soft splitter doesn't already exist right after the quote
        for start, close in quote_pairs:
            allow_split, split_type = cls._analyze_quote_pair_split_point(
                text, close, start
            )

            if allow_split and split_type == "soft":
                pos_after_close = close + 1

                # Check if there's a soft splitter right after the quote close
                # If so, skip adding a separate quote-based split point
                has_adjacent_soft_splitter = False
                if pos_after_close < len(text):
                    char_after = text[pos_after_close]
                    if char_after in {",", "，", ":", "："}:
                        has_adjacent_soft_splitter = True

                # Only add soft split from quote if there's no adjacent soft splitter
                if not has_adjacent_soft_splitter:
                    pos = close + 1
                    # Extend with additional closing symbols
                    while pos < len(text) and text[pos] in cls.CLOSERS:
                        pos += 1
                    if 0 < pos <= len(text) and pos not in points:
                        points.append(pos)

        return sorted(points)
