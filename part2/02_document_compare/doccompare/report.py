"""頂層報告：結構化欄位渲染表格 + LLM 敘事結論（改走 common 層的 complete）。

保留的既有修正：
    2.2-15  頁碼與風險表格「由結構化欄位在程式端渲染」，不經模型自由文字萃取，避免漏抄/幻覺。
    2.2-18  頂層報告採階層彙整（章節數過多時先分批產小節再合併），避免頂層單次輸入超標。
    2.2-1   報告附「未成功比對段落」清單（render_unmatched）與「未成功彙總章節」清單。

設計要點：整體結論以 common.complete 一次產生（同步 → asyncio.to_thread）。敘事失敗不
影響表格輸出。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from . import config, llm
from .reduce_stage import _fits, summarize_group

log = logging.getLogger(__name__)

_RISK_ICON = {"high": "🔴 高", "medium": "🟡 中", "low": "🟢 低", "none": "－"}


def _pp(a: Optional[int], b: Optional[int]) -> str:
    """頁碼區間渲染：p.12 / p.12–13 / －。"""
    if not a:
        return "－"
    return f"p.{a}" if not b or b == a else f"p.{a}–{b}"


def render_diff_table(summaries: List[Dict[str, Any]]) -> str:
    rows = [
        "| 章節 | A 版 (頁) | B 版 (頁) | 差異摘要 | 風險 |",
        "|------|-----------|-----------|----------|------|",
    ]
    for s in summaries:
        if "error" in s:
            continue
        for d in s.get("differences", []):
            diffs = "；".join(d.get("differences", [])) or d.get("verdict", "")
            rows.append(
                f"| {s.get('section', '?')} "
                f"| {_pp(d.get('page_a'), d.get('page_a_end'))} "
                f"| {_pp(d.get('page_b'), d.get('page_b_end'))} "
                f"| {diffs} "
                f"| {_RISK_ICON.get(d.get('risk', 'none'), '－')} |"
            )
    return "\n".join(rows)


def render_unmatched(failures: List[Dict[str, Any]]) -> str:
    if not failures:
        return ""
    lines = ["### 未成功比對段落（需重試 / 人工複核）"]
    for f in failures:
        lines.append(
            f"- 章節「{f.get('section') or '?'}」 "
            f"A頁 {f.get('page_a')} / B頁 {f.get('page_b')}："
            f"{f.get('reason', '解析或拒答失敗')}"
        )
    return "\n".join(lines)


def render_section_errors(summaries: List[Dict[str, Any]]) -> str:
    errs = [s for s in summaries if "error" in s]
    if not errs:
        return ""
    lines = ["### 未成功彙總章節（reduce 失敗，需人工複核）"]
    for s in errs:
        lines.append(f"- 章節「{s.get('section', '?')}」：{s.get('error')}")
    return "\n".join(lines)


async def _narrative(valid: List[Dict[str, Any]]) -> str:
    """以 common.complete 產生整體結論與重大風險敘述（不自行編造頁碼）。"""
    system = (
        "你是資深法務顧問。根據各章節『結構化差異摘要』寫出兩份合約的整體比較結論："
        "先一句總評，再點出 2–4 個最重大的風險差異（說明對甲方或乙方的影響）。"
        "語氣專業精簡，用繁體中文，勿自行編造頁碼（頁碼表格另由程式渲染），"
        "勿逐字複述所有條款，聚焦高/中風險項目。"
    )
    user = "各章節結構化摘要如下：\n" + json.dumps(valid, ensure_ascii=False)
    return await asyncio.to_thread(
        llm.complete, system, user, max_tokens=config.REPORT_MAX_TOKENS_CAP
    )


async def final_report(
    section_summaries: List[Dict[str, Any]], failures: List[Dict[str, Any]]
) -> str:
    """組出最終 Markdown 報告：敘事（LLM）+ 表格（結構化渲染）+ 未成功清單。"""
    valid = [s for s in section_summaries if "error" not in s]

    # 章節數過多 → 先把章節摘要階層彙整（複用 summarize_group），避免頂層單次超標（對應 2.2-18）。
    while valid and not _fits(json.dumps(valid, ensure_ascii=False)) and len(valid) > 2:
        half: List[Dict[str, Any]] = []
        for i in range(0, len(valid), 8):
            half.append(await summarize_group("（跨章）", valid[i : i + 8], merge=True))
        valid = [s for s in half if "error" not in s]

    table = render_diff_table(section_summaries)  # 表格一律由「原始章節摘要」的結構化欄位渲染

    try:
        narrative = (await _narrative(valid)).strip() or "（無整體結論）"
    except Exception as e:
        log.warning("final_report 敘事生成失敗：%r", e)
        narrative = f"（整體結論生成失敗：{type(e).__name__}: {e}；以下表格仍可參考。）"

    blocks = ["## 文件比較報告", narrative, "### 差異點總覽", table]
    sec_err = render_section_errors(section_summaries)
    if sec_err:
        blocks.append(sec_err)
    unmatched = render_unmatched(failures)
    if unmatched:
        blocks.append(unmatched)
    return "\n\n".join(b for b in blocks if b)


def structural_report(pairs) -> str:
    """離線 dry-run 報告：不呼叫 LLM，只依對齊結果列出配對/獨有與相似度、頁碼區間。

    供無 LLM key 時驗證解析＋切塊＋對齊三個階段是否運作。
    """
    matched = [(a, b, s) for a, b, s in pairs if a is not None and b is not None]
    a_only = [a for a, b, _ in pairs if b is None]
    b_only = [b for a, b, _ in pairs if a is None]

    lines = [
        "## 文件比較報告（dry-run，未經 LLM 分析）",
        "",
        f"- 配對段落：{len(matched)}　A 獨有：{len(a_only)}　B 獨有：{len(b_only)}",
        "",
        "### 配對段落（依相似度）",
        "| A 章節 | A 頁 | B 章節 | B 頁 | 相似度 |",
        "|--------|------|--------|------|--------|",
    ]
    for a, b, s in sorted(matched, key=lambda t: t[2], reverse=True):
        lines.append(
            f"| {a.section or '?'} | {_pp(a.page_start, a.page_end)} "
            f"| {b.section or '?'} | {_pp(b.page_start, b.page_end)} | {s:.3f} |"
        )
    if a_only:
        lines += ["", "### A 版獨有"]
        lines += [f"- {c.section or '?'}（{_pp(c.page_start, c.page_end)}）" for c in a_only]
    if b_only:
        lines += ["", "### B 版獨有"]
        lines += [f"- {c.section or '?'}（{_pp(c.page_start, c.page_end)}）" for c in b_only]
    return "\n".join(lines)
