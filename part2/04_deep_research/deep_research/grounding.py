# -*- coding: utf-8 -*-
"""可溯源與抗注入的工程手段：quote-grounding、引用驗證、外部內容隔離。

修正對照：
  * 2.4-17 wrap_untrusted 把抓取正文以 <external_content> 標為不可信資料。
  * 2.4-18 quote_supported 對整段引文壓縮空白後做最長公共子串比對（覆蓋 ≥85%）。
  * 2.4-37 validate_citations 以正規表達式擷取所有內文 [n]，驗證存在於 ref_map、
           移除越界編號 —— 內文引用不再只靠 LLM 自律。
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .config import QUOTE_MATCH_THRESHOLD

_WS_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _norm(s: str) -> str:
    return _WS_RE.sub("", (s or "").lower())


def quote_supported(quote: str, source_text: str, threshold: float = QUOTE_MATCH_THRESHOLD) -> bool:
    """整段引文（壓縮空白後）的最長公共子串覆蓋率是否達門檻。

    既降低「前段命中、其餘捏造」的誤放，也避免空白/換行差異造成的誤殺（2.4-18）。
    """
    q = _norm(quote)
    if len(q) < 8:                       # 太短的引文不足以佐證，直接拒絕
        return False
    s = _norm(source_text)
    if not s:
        return False
    matcher = SequenceMatcher(None, q, s, autojunk=False)
    block = matcher.find_longest_match(0, len(q), 0, len(s))
    return (block.size / len(q)) >= threshold


def wrap_untrusted(text: str) -> str:
    """把外部抓取內容標記為不可信資料，交給 Reader/Verifier/Synthesizer（2.4-17）。"""
    return (
        "<external_content untrusted=\"true\">\n"
        + (text or "")
        + "\n</external_content>"
    )


def validate_citations(report: str, valid_numbers):
    """擷取內文所有 [n]，移除不在 valid_numbers 的越界編號（2.4-37）。

    回傳 (清理後報告, 實際使用到的編號集合, 被移除的越界編號集合)。
    """
    valid = set(valid_numbers)
    used = set()
    invalid = set()

    def repl(m: "re.Match") -> str:
        n = int(m.group(1))
        if n in valid:
            used.add(n)
            return m.group(0)
        invalid.add(n)
        return ""  # 移除越界引用標記

    cleaned = _CITATION_RE.sub(repl, report)
    return cleaned, used, invalid
