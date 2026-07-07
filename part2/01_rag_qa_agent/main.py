"""
CLI 入口（互動版）
==================
載入設定與文件庫 → 建索引（Mistral embeddings） → 啟動自檢（確認「查無規定」拒答路徑可觸發） → 互動問答。

用法：
    python main.py
（需先設定環境變數 GROQ_API_KEY（LLM）與 MISTRAL_API_KEY（Embedding），或提供 .env。）

非互動、可直接跑完並印出結果的展示請改用：python run_demo.py
"""
from __future__ import annotations

import os
import sys

from config import DATA_DIR
from rag import KnowledgeBase, load_documents
from agent import HRAgent

# Windows 傳統主控台 (cp950) 直接 print 非 ASCII 可能拋 UnicodeEncodeError；
# Py3.7+ 改用 UTF-8 輸出，讓標榜「可直接執行」的範例不致崩潰（另建議使用支援 UTF-8 的終端）。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _print_event(ev_type: str, payload: dict) -> None:
    """把 run_tool_loop 的工具軌跡逐段印出（避免 emoji，相容 cp950 主控台）。"""
    if ev_type == "tool_call":
        print(f"\n  -> 呼叫 {payload['name']} {payload['arguments']}", flush=True)
    elif ev_type == "tool_result":
        r = payload["result"]
        preview = r if isinstance(r, str) else str(r)
        if len(preview) > 160:
            preview = preview[:160] + "…"
        print(f"     工具回傳：{preview}", flush=True)
    elif ev_type == "final":
        print("\n[回覆]", flush=True)


def main() -> None:
    for key in ("GROQ_API_KEY", "MISTRAL_API_KEY"):
        if not os.getenv(key):
            print(f"缺少環境變數 {key}；請於 .env 或系統環境變數設定後再執行。")
            return

    print("建立知識庫（以 Mistral mistral-embed 產生向量）…")
    kb = KnowledgeBase()
    kb.build(load_documents(DATA_DIR))
    print(f"已載入 {len(kb.chunks)} 個切塊。")

    # 啟動自檢：對已知 out-of-scope 問題檢查門檻能否過濾，確認「查無規定」拒答路徑可被觸發
    probe = "公司有提供員工宿舍或停車位嗎？"
    passed = kb.search([probe])
    print(
        f"[自檢] out-of-scope 探測「{probe}」通過門檻段落數={len(passed)}"
        f"（應為 0；若非 0 請調高 RERANK_MIN_SCORE 或以驗證集校準）"
    )

    agent = HRAgent(kb)
    print("\nHR 制度問答助理已就緒。輸入 exit 離開。")
    print("試問：『我 2021-03-15 到職，今年可以休幾天特休？』")
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        print("助理：", end="", flush=True)
        result = agent.ask(q, on_event=_print_event)
        print(result["text"])


if __name__ == "__main__":
    main()
