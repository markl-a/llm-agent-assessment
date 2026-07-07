# -*- coding: utf-8 -*-
"""非互動 Deep Research 小型示範（走 common 層：Groq LLM + Mistral Embedding）。

以「小預算」對一個研究問題跑完整管線（planner → searcher → reader → verifier →
synthesizer），並印出：
  1) 規劃出的子問題
  2) 搜尋到並抓取收錄的來源（含網域、可信度）
  3) 交叉驗證後的發現（statement / 支持 / 反駁 / 信心）
  4) 最終帶內文引用 [n] 的報告 + 程式生成的參考文獻 + 覆蓋度附錄
  5) 收尾指標

刻意用小設定（1-2 個子問題、每題少量來源、單一迭代）以控制 LLM 呼叫量與時間。
非互動：不使用 input()，可直接 `python run_demo.py` 跑完。

需求：GROQ_API_KEY（LLM）。MISTRAL_API_KEY 為選用（缺少時向量 rerank 自動略過）。
"""
from __future__ import annotations

import logging
import os
import sys

# 把 part2/ 加入路徑，讓 deep_research 內部可 `import common`。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_research import Budget, DuckDuckGoSearch, deep_research
from deep_research import config

QUESTION = "什麼是 RAG（檢索增強生成），它和 fine-tuning 有何不同？各自的適用情境為何？"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass


def _tune_for_demo() -> None:
    """下修預算相關設定，控制呼叫量與時間（小預算跑法）。"""
    config.MAX_ITERATIONS = 1          # 單一迭代
    config.MAX_SEARCHER_STEPS = 5      # searcher 工具迴圈上限
    config.MAX_RESULTS_PER_SEARCH = 4  # 每次搜尋取回較少候選
    config.MAX_SOURCES_PER_SUBQ = 2    # 每子問題最多收 2 個來源
    config.SYNTH_MAX_TOKENS = 1800     # 報告長度上限（控制時間）


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    _load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        print("需要 GROQ_API_KEY 才能實際呼叫 LLM。請 `export GROQ_API_KEY=...`（或設定 .env）後重跑。",
              file=sys.stderr)
        return 2
    if not os.getenv("MISTRAL_API_KEY"):
        print("提示：未設定 MISTRAL_API_KEY，搜尋結果的向量 rerank 會自動略過（不影響流程）。",
              file=sys.stderr)

    _tune_for_demo()

    _hr(f"Deep Research 示範｜研究問題：{QUESTION}")
    print(f"LLM 供應商：{provider}（模型由 LLM_MODEL 覆寫，預設 llama-3.3-70b-versatile）")
    print("搜尋：DuckDuckGo（免金鑰）｜Embedding rerank："
          + ("啟用（Mistral）" if os.getenv("MISTRAL_API_KEY") else "略過（缺 MISTRAL_API_KEY）"))

    with DuckDuckGoSearch() as search:
        result = deep_research(
            QUESTION,
            search=search,
            budget=Budget(total=200_000),   # 小預算（估算加權 token）
            max_subquestions=2,             # 只保留 2 個子問題
        )

    state = result["state"]

    _hr("1) 規劃出的子問題")
    for s in state.subquestions:
        mark = "已覆蓋" if s.covered else "覆蓋不完整"
        print(f"  #{s.id} [{mark} conf={s.confidence:.2f}] {s.text}")

    _hr("2) 搜尋到並抓取收錄的來源")
    if not state.sources:
        print("  （未收錄任何來源；可能是 DuckDuckGo 暫時限流或網路問題）")
    for sid, src in sorted(state.sources.items()):
        print(f"  #{sid} {src.title[:70]}")
        print(f"       {src.url}")
        print(f"       網域={src.registered_domain}｜可信度={src.credibility}")

    _hr("3) 交叉驗證後的發現（findings）")
    if not state.findings:
        print("  （無已驗證發現）")
    for f in state.findings:
        flag = "（有反面證據）" if f.contradicted else ""
        print(f"  [信心 {f.confidence:.2f}]{flag} {f.statement}")
        print(f"       支持來源 id={f.supporting}｜反駁來源 id={f.conflicting}")

    _hr("4) 最終報告（含內文引用 / 參考文獻 / 覆蓋度附錄）")
    print(result["report"])

    _hr("5) 指標")
    print(result["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
