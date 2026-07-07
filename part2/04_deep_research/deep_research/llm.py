# -*- coding: utf-8 -*-
"""LLM / Embedding 介面 —— 共用 provider 抽象層（part2/common）的薄封裝。

本檔為 OpenAI 相容共用層之上的薄封裝：

  * LLM：預設 Groq llama-3.3-70b-versatile。
      - 規劃 / 逐來源閱讀 / 交叉驗證 / 缺口分析 / 報告彙整 → common.complete（一次性文字）
      - 搜尋 agent（web_search / fetch_page / done）→ common.run_tool_loop（自動工具迴圈）
  * Embedding：預設 Mistral mistral-embed（1024 維），用於搜尋結果的輕量向量排序
      （以 API embeddings 取代本地 cross-encoder reranker：更輕量、可降級）。

供應商 / 模型一律由環境變數決定（LLM_PROVIDER / LLM_MODEL / EMBED_PROVIDER /
EMBED_MODEL），金鑰走環境變數（GROQ_API_KEY / MISTRAL_API_KEY…），不在程式內寫死。

結構化輸出：OpenAI 相容 complete 不保證支援 json_schema 強制格式，故改以「system 提示
要求只輸出 JSON ＋ 穩健解析（去 ``` 圍欄、擷取首個 {...} 區塊）」達成（見 call_json）。

預算：common 層不回傳 token 用量，改以字元數估算（約 4 chars/token）計入 Budget，
用於「預算耗盡即強制進入報告階段」的收斂控制。
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys

# 讓 `import common` 可用：把 part2/ 加入模組搜尋路徑。
# 本檔位於 part2/04_deep_research/deep_research/llm.py → 上溯三層即 part2/。
_PART2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PART2 not in sys.path:
    sys.path.insert(0, _PART2)

from common import complete, run_tool_loop, embed_texts, cosine_sim  # noqa: E402

from .models import Budget, Usage  # noqa: E402

log = logging.getLogger("deep_research.llm")


class LLMError(Exception):
    pass


# ---------------------------------------------------------------------------
# 預算估算（common 層不回傳 usage）
# ---------------------------------------------------------------------------
def _estimate_usage(prompt: str, completion: str) -> Usage:
    """以字元數粗估 token 用量（約 4 chars/token），用於驅動預算控制。"""
    return Usage(input_tokens=len(prompt or "") // 4, output_tokens=len(completion or "") // 4)


# ---------------------------------------------------------------------------
# 一次性文字回覆
# ---------------------------------------------------------------------------
def complete_text(budget: Budget, system, user, *, max_tokens: int = 1024,
                  temperature: float = 0.0) -> str:
    """走 common.complete 取得一次性文字，並把估算用量計入預算。"""
    text = complete(system, user, temperature=temperature, max_tokens=max_tokens)
    budget.add(_estimate_usage((system or "") + user, text))
    return text


# ---------------------------------------------------------------------------
# 穩健 JSON 解析（OpenAI 相容端點不保證 json_schema 強制格式，改以文字解析）
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_JSON_DIRECTIVE = "\n\n請只輸出一個 JSON 物件本身，禁止任何前後說明文字，禁止 markdown 圍欄。"


def _json_candidates(t: str):
    """依序產生可嘗試解析的候選字串：整段 → 去 ``` 圍欄 → 首個 {...} 區塊。"""
    yield t
    m = _FENCE_RE.search(t)
    if m:
        yield m.group(1)
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        yield t[start:end + 1]


def _extract_json(text: str):
    if not text:
        return None
    for candidate in _json_candidates(text.strip()):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def call_json(budget: Budget, system, user, *, attempts: int = 2,
              max_tokens: int = 1024) -> dict:
    """呼叫 LLM 並解析結構化 JSON，失敗時重試。

    每次呼叫（含失敗的嘗試）都把估算用量計入預算，確保成本統計不失真。
    回傳 dict；連續失敗則丟出 LLMError（呼叫端多以 try/except 降級跳過）。
    """
    last_err: Exception | None = None
    for i in range(attempts):
        hint = _JSON_DIRECTIVE if i == 0 else (
            _JSON_DIRECTIVE + "\n（上一次輸出無法解析為合法 JSON，請務必只輸出合法 JSON。）")
        text = complete(system, user + hint, temperature=0.0, max_tokens=max_tokens)
        budget.add(_estimate_usage((system or "") + user, text))
        data = _extract_json(text)
        if isinstance(data, dict):
            return data
        last_err = LLMError(f"無法解析為 JSON：{(text or '')[:180]!r}")
        log.warning("結構化解析失敗（嘗試 %d/%d）", i + 1, attempts)
    raise last_err or LLMError("call_json 失敗")


# ---------------------------------------------------------------------------
# 搜尋 agent 迴圈（tool-calling）：包一層 common.run_tool_loop
# ---------------------------------------------------------------------------
def _rough_text(messages) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        for tc in (m.get("tool_calls") or []):
            parts.append(((tc.get("function") or {}).get("arguments")) or "")
    return "".join(parts)


def tool_loop(budget: Budget, system, messages, tools, tool_impls, *,
              max_rounds: int = 6, max_tokens: int = 1024, on_event=None) -> dict:
    """走 common.run_tool_loop 執行自動工具迴圈，回傳其結果並把估算用量計入預算。

    common.run_tool_loop 會就地更新 messages（可當記憶），並自動：呼叫模型 → 依
    tool_impls 執行工具 → 以 role=tool 回饋 → 直到模型不再呼叫工具或達 max_rounds。
    """
    result = run_tool_loop(system, messages, tools, tool_impls,
                           max_rounds=max_rounds, temperature=0.0,
                           max_tokens=max_tokens, on_event=on_event)
    budget.add(_estimate_usage(_rough_text(result.get("messages", [])), ""))
    return result


# ---------------------------------------------------------------------------
# Embedding 輔助（走 Mistral API）
# ---------------------------------------------------------------------------
def embed_rerank(query: str, hits, top_k=None):
    """用向量相似度對搜尋結果重新排序（輕量 rerank）。

    以 Mistral embeddings 對 query 與各結果的 title+snippet 求 cosine 相似度後排序。
    缺 MISTRAL_API_KEY 或任何錯誤都會靜默降級為原順序，確保不阻斷主流程。
    （以 API embeddings 取代本地 cross-encoder reranker：更輕量、可降級。）
    """
    if not hits:
        return hits
    try:
        docs = [f"{h.get('title', '')} {h.get('snippet', '')}".strip() or h.get("url", "")
                for h in hits]
        vecs = embed_texts([query] + docs)
        qv, dvs = vecs[0], vecs[1:]
        scored = sorted(zip(hits, dvs), key=lambda p: cosine_sim(qv, p[1]), reverse=True)
        ranked = [h for h, _ in scored]
        return ranked[:top_k] if top_k else ranked
    except Exception as e:  # noqa: BLE001 — rerank 只是加分項，失敗即降級
        log.info("向量 rerank 略過（降級為原順序）：%s", e)
        return hits
