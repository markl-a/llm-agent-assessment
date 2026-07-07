# -*- coding: utf-8 -*-
"""Deep Research Agent 套件。

對研究問題做多來源、多步驟研究並產出有引用的報告：
planner → searcher → reader → verifier → synthesizer。

LLM / Embedding 呼叫改接共用 provider 抽象層（part2/common）：
  * LLM 預設 Groq llama-3.3-70b-versatile（complete / run_tool_loop）。
  * Embedding 預設 Mistral mistral-embed（搜尋結果輕量向量排序）。
  * 網路搜尋預設 DuckDuckGo（免金鑰，透過 SearchProvider 抽象可替換）。
"""
from .models import Budget, ResearchState
from .orchestrator import (
    compute_confidence,
    deep_research,
    load_checkpoint,
    save_checkpoint,
    synthesize,
)
from .search import DuckDuckGoSearch, SearchProvider

__all__ = [
    "Budget",
    "ResearchState",
    "deep_research",
    "compute_confidence",
    "synthesize",
    "save_checkpoint",
    "load_checkpoint",
    "SearchProvider",
    "DuckDuckGoSearch",
]
