"""
非互動示範（run_demo）
=======================
可直接 `python run_demo.py` 跑完，不需任何輸入。流程：

  1. 建知識庫（Mistral embeddings）、印出切塊數。
  2. out-of-scope 自檢：對「員工宿舍 / 停車位」探測，確認「查無規定」拒答路徑可觸發（通過段落數應為 0）。
  3. 內建兩個問題連續提問（共用同一 HRAgent，展示對話記憶）：
       Q1  觸發特休試算：檢索公司累進表 → 帶入 calculate_annual_leave → 對照勞基法。
       Q2  追問（帶指代）：展示對話記憶 + 多查詢改寫解析指代 + 另一份文件來源。
     每題印出：檢索到的來源、工具呼叫軌跡（run_tool_loop 事件）、最終「帶引用」的回答。
  4. Streaming 展示：用 common.stream_complete 逐字串流一段白話小結。

供應商由共用層 part2/common 以環境變數選擇（預設 LLM=Groq、Embedding=Mistral）；
金鑰走環境變數（GROQ_API_KEY / MISTRAL_API_KEY），不寫死於程式。
"""
from __future__ import annotations

import os
import sys

# 把 part2 加入路徑後即可 `from common import ...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import stream_complete  # noqa: E402

from config import DATA_DIR  # noqa: E402
from rag import KnowledgeBase, load_documents  # noqa: E402
from agent import HRAgent  # noqa: E402

# Windows 傳統主控台 (cp950) 直接 print 非 ASCII 可能拋 UnicodeEncodeError；改用 UTF-8。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEMO_QUESTIONS = [
    # Q1：會觸發「檢索 → 帶入累進表 → calculate_annual_leave」的多工具資料互用
    "我 2021-03-15 到職，以 2026-07-06 為基準，今年可以休幾天特休？公司制度有比勞基法好嗎？",
    # Q2：帶指代的追問，展示對話記憶 + 多查詢改寫 + 另一份文件來源
    "那如果我出差，住宿一天最多可以報多少？",
]


def _banner(text: str) -> None:
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def make_event_printer(agent: HRAgent):
    """回傳 run_tool_loop 的 on_event handler：逐段印出工具軌跡，並在檢索後印出命中來源。"""

    def on_event(ev_type: str, payload: dict) -> None:
        if ev_type == "tool_call":
            print(f"\n  [工具呼叫] {payload['name']}  參數={payload['arguments']}", flush=True)
        elif ev_type == "tool_result":
            name = payload["name"]
            if name == "search_hr_knowledge_base":
                if agent.last_hits:
                    print("  [檢索到的來源]", flush=True)
                    for h in agent.last_hits:
                        snippet = h.text[:48].replace("\n", " ")
                        print(f"     - [{h.id}] {h.doc_title}（{h.source}）：{snippet}…", flush=True)
                else:
                    print("  [檢索到的來源] （無 —— 低於相關度門檻，將誠實拒答）", flush=True)
            else:
                r = payload["result"]
                preview = r if isinstance(r, str) else str(r)
                print(f"  [工具回傳] {preview}", flush=True)
        elif ev_type == "final":
            print("\n  [產生最終回答…]", flush=True)

    return on_event


def main() -> None:
    missing = [k for k in ("GROQ_API_KEY", "MISTRAL_API_KEY") if not os.getenv(k)]
    if missing:
        print("=" * 68)
        print("無法執行 demo：缺少必要環境變數 " + ", ".join(missing))
        print("-" * 68)
        print("本 demo 需要：")
        print("  GROQ_API_KEY     —— LLM（Groq，llama-3.3-70b-versatile）")
        print("  MISTRAL_API_KEY  —— Embedding（Mistral，mistral-embed）")
        print("請於專案根目錄建立 .env（可參考 .env.example）或設定系統環境變數後再執行：")
        print("  python run_demo.py")
        print("=" * 68)
        sys.exit(1)

    _banner("步驟 1／3：建立知識庫（Mistral mistral-embed 向量化）")
    kb = KnowledgeBase()
    kb.build(load_documents(DATA_DIR))
    print(f"已載入文件並切塊：共 {len(kb.chunks)} 個切塊。")
    for c in kb.chunks:
        print(f"  - [{c.id}] {c.doc_title}（{c.source}）")

    _banner("步驟 2／3：out-of-scope 自檢（確認『查無規定』拒答路徑可觸發）")
    probe = "公司有提供員工宿舍或停車位嗎？"
    passed = kb.search([probe])
    print(f"探測問題：「{probe}」")
    print(
        f"通過相關度門檻的段落數 = {len(passed)}"
        f"（應為 0；代表 LLM 重排把無關段落濾掉，拒答路徑確實可被觸發）"
    )

    _banner("步驟 3／3：連續問答（同一 HRAgent＝具對話記憶）")
    agent = HRAgent(kb)
    on_event = make_event_printer(agent)
    last_answer = ""
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n──────── 問題 {i} ────────")
        print(f"使用者：{q}")
        result = agent.ask(q, on_event=on_event)
        last_answer = result["text"]
        print(f"\n助理（工具往返 {result['rounds']} 回合，帶引用）：")
        print(last_answer)

    _banner("Streaming 展示（common.stream_complete 逐字串流）")
    print("請 AI 用一句白話總結上面的特休結論：\n")
    try:
        for piece in stream_complete(
            "你是 HR 助理，用一句繁體中文白話說明，不要加來源標註。",
            "請用一句話總結：這位 2021-03-15 到職的員工，今年公司特休天數與是否優於勞基法。\n"
            f"參考先前結論：{last_answer[:600]}",
            max_tokens=160,
        ):
            print(piece, end="", flush=True)
        print()
    except Exception as e:  # noqa: BLE001 — 串流示範失敗不影響主流程
        print(f"（串流示範略過：{e}）")

    _banner("Demo 結束")
    print("重點展示：Mistral 向量檢索＋LLM 重排（含誠實拒答門檻）、多查詢改寫、")
    print("多工具資料互用（累進表帶入特休試算）、帶出處引用、對話記憶、串流。")


if __name__ == "__main__":
    main()
