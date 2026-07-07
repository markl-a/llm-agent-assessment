"""資料模型：Chunk 與各層 Structured Output schema。

- Chunk：解析階段的切塊，保留頁碼「區間」與章節 metadata（對應 2.2-11 的頁碼精確度）。
- PairDiff / SectionSummary：map / reduce 兩層以 Pydantic schema 約束模型解碼，
  確保輸出可被機器彙總（Structured Output）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pydantic import BaseModel


@dataclass
class Chunk:
    """一個切塊。

    刻意不保留 draft 版本中恆為 0、從未被讀取的 idx 欄位（對應 2.2-36 死碼移除）。
    對齊全程以 enumerate 位置索引與 id(chunk) 運作，不需要序號。
    """

    doc_id: str
    text: str
    page_start: int          # 1-indexed，本塊主要內容起始頁
    page_end: int            # 本塊結束頁；報告以頁碼「區間」呈現
    section: Optional[str]   # 由標題結構推得（None = 未分章）
    embedding: Optional[List[float]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# map 層：逐對段落結構化差異
# ---------------------------------------------------------------------------
class PairDiff(BaseModel):
    """單一對段落的差異分析結果（由 llm.complete_json 產生並以本 schema 驗證）。"""

    verdict: Literal["identical", "modified", "a_only", "b_only"]
    same_points: List[str]
    differences: List[str]
    risk: Literal["none", "low", "medium", "high"]
    risk_note: str


# ---------------------------------------------------------------------------
# reduce 層：章節摘要（保留 page_*/risk 等結構化欄位，避免退化成自由文字；對應 2.2-15）
# ---------------------------------------------------------------------------
class DiffItem(BaseModel):
    verdict: str
    differences: List[str]
    risk: Literal["none", "low", "medium", "high"]
    page_a: Optional[int] = None
    page_a_end: Optional[int] = None
    page_b: Optional[int] = None
    page_b_end: Optional[int] = None


class SectionSummary(BaseModel):
    section: str
    same_points: List[str]
    differences: List[DiffItem]
    risk: Literal["none", "low", "medium", "high"]
    risk_note: str
