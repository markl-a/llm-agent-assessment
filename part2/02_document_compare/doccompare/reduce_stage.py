"""REDUCE：階層式 map-reduce 彙總（改走 common 層的 complete_json，Structured Output）。

保留的既有修正：
    2.2-15  reduce 也用 Structured Output 保留 section/differences/risk/page_* 欄位，
            頁碼與風險等級不退化成自由文字。
    2.2-8   summarize_group 解析/驗證失敗不靜默產生空摘要，保留可辨識標記供複核。
    2.2-10  為 reduce 另設 asyncio.Semaphore 並包住呼叫，避免瞬間並發過多。
    2.2-32  群組是否超預算改以「保守字元上界估算」遞迴切分，而非以筆數粗估。
    2.2-38  合併層（merge）用區別於葉層（leaf）的提示，避免「提示與實際餵入資料語意不符」。

設計要點：token 計數以字元上界估算；LLM 呼叫以 asyncio.to_thread 包住同步的
common.complete_json，維持並發。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

from . import config, llm
from .models import SectionSummary

log = logging.getLogger(__name__)

_REDUCE_SEM = asyncio.Semaphore(config.REDUCE_CONCURRENCY)

REDUCE_SYSTEM = (
    "你是法律／商務文件比對的彙整助手。請把輸入的『逐段差異 / 子摘要』彙整為單一章節摘要，"
    "務必保留每項差異的頁碼（page_a/page_a_end/page_b/page_b_end 原樣帶入 differences[]，"
    "不得自行編造頁碼）與最高風險等級。differences[] 每項的 differences 欄位放具體差異短句、"
    "verdict 沿用來源、risk 取該項最高。section 用來源章節名。同一章節的最高風險寫入頂層 risk。"
)

LEAF_PROMPT = (
    "以下 <items> 是本章節的『逐段差異 JSON』（含 page_a/page_b/risk），"
    "請彙整為單一章節摘要，保留關鍵頁碼區間與最高風險等級。"
)
MERGE_PROMPT = (
    "以下 <items> 是本章節數個『子摘要』（非逐段差異），請合併為單一章節摘要，"
    "保留關鍵頁碼區間與最高風險等級。"
)


def _fits(payload: str) -> bool:
    """判斷單次 reduce 輸入是否在預算內（以中文約 1 token/字元的保守字元上界估算）。"""
    return llm.estimate_tokens(payload) <= config.REDUCE_TOKEN_BUDGET


def _is_map_result(item: Dict[str, Any]) -> bool:
    """判斷是否為「葉層」的單筆 map 結果（含 verdict 與 str 陣列 differences），
    以區別於已彙總的子摘要（其 differences 為 dict 陣列、且無頂層 verdict）。"""
    return "verdict" in item and "risk_note" in item


def _passthrough_summary(section: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """單一 map 結果 → SectionSummary（不呼叫 LLM）。

    此為省成本且更可靠的捷徑：當某章節只有一筆逐段差異時，該筆本身已是結構化輸出，
    無須再讓 LLM 重述——直接把 verdict/differences/risk 與「頁碼」原樣帶入 SectionSummary，
    頁碼因此完全不依賴模型轉抄，避免漏抄/幻覺（保留 2.2-15 的精神）。多筆的章節仍走 LLM
    彙總（見 summarize_group），那才是 reduce 真正發揮判斷力之處。
    """
    diff_item = {
        "verdict": item.get("verdict", ""),
        "differences": list(item.get("differences", [])),
        "risk": item.get("risk", "none"),
        "page_a": item.get("page_a"),
        "page_a_end": item.get("page_a_end"),
        "page_b": item.get("page_b"),
        "page_b_end": item.get("page_b_end"),
    }
    return {
        "section": section,
        "same_points": list(item.get("same_points", [])),
        "differences": [diff_item],
        "risk": item.get("risk", "none"),
        "risk_note": item.get("risk_note", ""),
    }


async def summarize_group(
    section: str, items: List[Dict[str, Any]], merge: bool = False
) -> Dict[str, Any]:
    """彙總單一群組；群組過大則依預算遞迴切半 → 各自彙總 → 合併。

    單筆 map 結果採本地 passthrough（不呼叫 LLM），大幅降低 LLM 呼叫次數與 token；
    多筆才進 LLM 彙總。
    """
    if not merge and len(items) == 1 and _is_map_result(items[0]):
        return _passthrough_summary(section, items[0])

    payload = json.dumps(items, ensure_ascii=False)

    if not _fits(payload) and len(items) > 1:
        mid = len(items) // 2
        left, right = await asyncio.gather(
            summarize_group(section, items[:mid], merge),
            summarize_group(section, items[mid:], merge),
        )
        # 進入合併層：items 已是子摘要 dict，改用 MERGE_PROMPT（對應 2.2-38）。
        return await summarize_group(section, [left, right], merge=True)

    prompt = (MERGE_PROMPT if merge else LEAF_PROMPT) + f"\n<items>\n{payload}\n</items>"
    try:
        await llm.llm_limiter.acquire(
            llm.estimate_tokens(prompt) + llm.estimate_tokens(REDUCE_SYSTEM)
        )
        async with _REDUCE_SEM:  # reduce 專屬限流
            parsed: SectionSummary = await asyncio.to_thread(
                llm.complete_json,
                REDUCE_SYSTEM,
                prompt,
                SectionSummary,
                example=llm.SECTIONSUMMARY_EXAMPLE,
                max_tokens=config.REDUCE_MAX_TOKENS,
            )
        out = parsed.model_dump()
        # 章節名以來源為準（避免模型改寫）。
        if not out.get("section"):
            out["section"] = section
        return out
    except Exception as e:  # 例外不擴散，保留可辨識標記（對應 2.2-8）
        log.warning("summarize_group 失敗（section=%s）：%r", section, e)
        return {"section": section, "error": f"{type(e).__name__}: {e}"}


async def run_reduce(map_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """依章節 group，餵『差異 JSON』而非原文；gather 帶 return_exceptions 保護。"""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in map_results:
        groups[r.get("section") or "（未分章）"].append(r)

    tasks = [summarize_group(s, items) for s, items in groups.items()]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    sections: List[Dict[str, Any]] = []
    for (sec, _), r in zip(groups.items(), gathered):
        if isinstance(r, Exception):
            sections.append({"section": sec, "error": repr(r)})
        else:
            sections.append(r)
    return sections
