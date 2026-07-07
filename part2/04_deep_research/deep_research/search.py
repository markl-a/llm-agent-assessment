# -*- coding: utf-8 -*-
"""搜尋供應商抽象與安全的網頁抓取/擷取。

本次無搜尋 API 金鑰，預設改用 DuckDuckGo 免金鑰搜尋（pip 套件 `ddgs`）。搜尋介面
刻意抽象化（SearchProvider）：要改接其他供應商（Tavily / Bing / SerpAPI…）只需
實作 `SearchProvider.search(query, max_results)` 回傳 [{'title','url','snippet'}, ...]，
不需改動其他程式。

安全性保留：
  * fetch_and_extract 逐跳驗證 redirect、限制 http/https、設超時與回應大小上限（SSRF 防護）。
  * 從頁面 metadata / <title> 擷取真實標題（避免參考清單出現「URL — URL」）。
  * 搜尋對網路例外重試 / 降級容錯空結果，不把錯誤偽裝成「沒有結果」。

正文擷取：優先 trafilatura，退回 beautifulsoup4（擇一即可；兩者皆缺時退回極簡去標籤）。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import requests

from . import config
from .net import assert_safe_url, resolve_redirect, UnsafeURLError

log = logging.getLogger("deep_research.search")

# 選用相依：優先 trafilatura，退回 BeautifulSoup，再退回極簡去標籤。
try:  # pragma: no cover - 視安裝環境而定
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except Exception:  # pragma: no cover
    trafilatura = None
    _HAS_TRAFILATURA = False

try:  # pragma: no cover
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:  # pragma: no cover
    BeautifulSoup = None
    _HAS_BS4 = False


class SearchError(Exception):
    pass


class SearchProvider(ABC):
    """搜尋供應商介面。"""

    @abstractmethod
    def search(self, query: str, max_results: int) -> List[dict]:
        """回傳 [{'title', 'url', 'snippet'}, ...]；查無結果回傳空 list。"""

    def close(self) -> None:  # 預設 no-op
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class DuckDuckGoSearch(SearchProvider):
    """DuckDuckGo 免金鑰搜尋（透過 `ddgs` 套件）。

    無需任何 API 金鑰。對暫時性錯誤重試、退避；最終失敗降級為空結果，
    讓上游繼續處理其他子問題（而非整個流程崩潰）。
    """

    def __init__(self, max_retries: int = 3, region: str = "wt-wt"):
        try:
            from ddgs import DDGS  # type: ignore
        except Exception as e:  # pragma: no cover - 視安裝環境而定
            raise SearchError("未安裝 ddgs，請先 `pip install ddgs`") from e
        self._DDGS = DDGS
        self.max_retries = max_retries
        self.region = region

    def search(self, query: str, max_results: int) -> List[dict]:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with self._DDGS() as ddgs:
                    raw = ddgs.text(query, region=self.region, max_results=max_results)
                out: List[dict] = []
                for item in (raw or []):
                    url = item.get("href") or item.get("url") or item.get("link")
                    if not url:
                        continue
                    out.append({
                        "title": (item.get("title") or "").strip(),
                        "url": url,
                        "snippet": (item.get("body") or item.get("snippet") or "")[:400],
                    })
                return out
            except Exception as e:  # noqa: BLE001 — DDG 限流/網路波動時重試後降級
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        log.warning("DuckDuckGo 搜尋在 %d 次嘗試後失敗，降級為空結果：%s",
                    self.max_retries, last_err)
        return []

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 網頁抓取與正文擷取
# ---------------------------------------------------------------------------
class FetchError(Exception):
    pass


def _charset_from_ctype(ctype: str):
    """從 content-type 標頭取 charset（例如 'text/html; charset=utf-8' → 'utf-8'）。"""
    ctype = (ctype or "").lower()
    if "charset=" not in ctype:
        return None
    enc = ctype.split("charset=", 1)[1].split(";")[0].strip().strip('"\'')
    return enc or None


def _extract(html: str, url: str) -> Tuple[str, str]:
    """回傳 (正文, 標題)。"""
    title = ""
    text = ""
    if _HAS_TRAFILATURA:
        try:
            text = trafilatura.extract(html, url=url, include_comments=False,
                                       include_tables=False) or ""
            meta = trafilatura.extract_metadata(html)
            if meta and getattr(meta, "title", None):
                title = meta.title or ""
        except Exception:
            text = ""
    if not text and _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        if soup.title and soup.title.string:
            title = title or soup.title.string.strip()
        text = "\n".join(
            s for s in (line.strip() for line in soup.get_text("\n").splitlines()) if s
        )
    if not text:  # 最後退路：極簡去標籤
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
    return text, (title or "").strip()


def fetch_and_extract(
    url: str,
    session: Optional[requests.Session] = None,
    allowlist=None,
) -> Tuple[str, str]:
    """安全抓取單一 URL 並擷取正文，回傳 (正文, 標題)。

    逐跳驗證 redirect 目標（防 SSRF），限制大小與逾時；非 HTML/文字內容直接拒絕。
    """
    sess = session or requests.Session()
    headers = {"User-Agent": config.USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    current = url
    for _ in range(config.FETCH_MAX_REDIRECTS + 1):
        assert_safe_url(current, allowlist)  # 每一跳都驗證
        resp = sess.get(
            current,
            headers=headers,
            timeout=config.FETCH_TIMEOUT,
            stream=True,
            allow_redirects=False,
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("location")
            resp.close()
            if not loc:
                raise FetchError("redirect 缺少 Location")
            current = resolve_redirect(current, loc)
            continue
        try:
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").lower()
            if not any(t in ctype for t in ("text/html", "application/xhtml", "text/plain")):
                raise FetchError(f"不支援的內容型別：{ctype or 'unknown'}")
            total = 0
            chunks = []
            for chunk in resp.iter_content(8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > config.FETCH_MAX_BYTES:
                    break  # 大小上限保護
                chunks.append(chunk)
            raw = b"".join(chunks)
            # 編碼判定：優先 content-type 的 charset；缺少時預設 utf-8。
            # 注意不要用 resp.encoding —— HTTP 規範下 requests 對未宣告 charset 的 text/*
            # 會退回 ISO-8859-1，導致 UTF-8 中文頁面變成亂碼。
            encoding = _charset_from_ctype(ctype) or "utf-8"
        finally:
            resp.close()
        html = raw.decode(encoding, errors="replace")
        return _extract(html, current)
    raise FetchError("redirect 次數過多")
