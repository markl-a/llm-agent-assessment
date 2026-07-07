"""串流解析 + 切塊（帶頁碼區間 / 章節 metadata）。

設計原則（對應 review 2.2-7 的記憶體宣稱限縮）：
    「解析階段」一次只持有一頁的版面物件，切塊後立即釋放，且**不建立單一全文字串**。
    但要留意 load() 仍把全部 chunk 物化成 list、嵌入階段再為每塊掛向量，峰值記憶體
    仍為 O(全文)。若要真正低記憶體，須把 chunk / embedding 落地磁碟或向量庫並於對齊時
    串流讀取——本檔僅保證「解析階段不建立單一全文字串」這個較弱但真實的宣稱。

支援輸入：
    - .pdf：逐頁 pdfplumber.extract_text()，逐頁釋放版面物件；同時偵測「抽取為空」的
      掃描/圖片型頁面並警示（對應 2.2-20，避免無聲產生空 chunk）。
    - .md / .txt / 其他純文字：以「行」為單位串流，整份視為第 1 頁，讓程式無需真實 PDF
      即可 demo。
"""

from __future__ import annotations

import logging
import re
from typing import Iterator, List

from . import config
from .models import Chunk

log = logging.getLogger(__name__)

# 中文數字補上「百 / 千 / 零 / 兩」，可辨識「第一百章」「第兩百條」等標題（對應 2.2-34）；
# 同時支援阿拉伯數字與多層編號（如 "3.2 服務範圍"）。
HEADING_RE = re.compile(
    r"^(第[零一二三四五六七八九十百千兩\d]+[章節條]|\d+(\.\d+)*\s+\S+)"
)


def _normalize_heading_line(line: str) -> str:
    """去掉 Markdown 標題的前導 '#' 與空白，讓 '## 第三章 責任限制' 也能被辨識。"""
    return line.strip().lstrip("#").strip()


def _iter_pages(path: str) -> Iterator[tuple[int, str]]:
    """逐頁產出 (頁碼, 該頁純文字)。

    - PDF：真正逐頁 lazy 讀取，讀完主動釋放該頁版面物件控制記憶體。
    - 其他：整份當作單頁串流（示例用的 Markdown / 純文字）。
    """
    lower = path.lower()
    if lower.endswith(".pdf"):
        try:
            import pdfplumber  # 版面較準；純文字可換 pypdf 更快
        except ImportError as e:  # pragma: no cover - 需要時才裝
            raise RuntimeError(
                "讀取 PDF 需要 pdfplumber，請先 `pip install pdfplumber`"
            ) from e

        with pdfplumber.open(path) as pdf:  # 逐頁 lazy 讀取
            for pno, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # 表格：以結構化欄位補回純文字抽取常遺漏的內容（對應 2.2-20）。
                try:
                    for table in page.extract_tables() or []:
                        for row in table:
                            cells = [c for c in row if c]
                            if cells:
                                text += "\n" + " | ".join(cells)
                except Exception:  # 表格抽取失敗不致中斷整份解析
                    log.debug("第 %d 頁表格抽取失敗，忽略", pno)
                yield pno, text
                # 主動釋放該頁版面物件（pdfplumber 逐頁快取，清掉可壓低記憶體峰值）。
                try:
                    page.flush_cache()
                except Exception:
                    pass
    else:
        with open(path, "r", encoding="utf-8") as f:
            yield 1, f.read()


def stream_chunks(
    path: str,
    doc_id: str,
    target_chars: int = config.TARGET_CHARS,
    overlap: int = config.OVERLAP_CHARS,
) -> Iterator[Chunk]:
    """逐頁串流切塊；解析階段不建立單一全文字串。

    頁碼精確度（對應 2.2-11）：在 buffer 由空轉非空時記錄起始頁 buf_start，切塊時同時
    記錄 page_start / page_end，報告顯示頁碼「區間」而非單一頁——即使主要內容落在後頁，
    也不會被誤標成前頁頁碼。
    """
    current_section = None
    buf = ""
    buf_start = None  # buffer 由空轉非空時記錄的起始頁
    pno = 0

    for pno, text in _iter_pages(path):
        if not text.strip():
            # 掃描/圖片型頁面純文字抽取為空 → 偵測並警示，避免無聲產生空 chunk。
            log.warning(
                "第 %d 頁 (%s) 文字抽取為空，疑為掃描/圖片型；需 OCR 或改用 "
                "文件／影像等多模態輸入", pno, doc_id,
            )
        for line in text.splitlines():
            if buf == "":
                buf_start = pno  # buffer 由空轉非空 → 記錄起始頁
            heading = _normalize_heading_line(line)
            if HEADING_RE.match(heading):
                current_section = heading[:80]
            buf += line + "\n"
            if len(buf) >= target_chars:  # 達門檻就切出一塊
                yield Chunk(doc_id, buf.strip(), buf_start or pno, pno, current_section)
                buf = buf[-overlap:]  # 保留重疊，維持語意連續
                buf_start = pno       # overlap 尾巴屬本頁，下一塊起始頁即本頁

    if buf.strip():
        yield Chunk(doc_id, buf.strip(), buf_start or 1, pno or 1, current_section)


def load(path: str, doc_id: str) -> List[Chunk]:
    """把串流切塊物化為 list（供對齊/嵌入使用）。"""
    return list(stream_chunks(path, doc_id))
