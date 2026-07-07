"""run_demo.py —— 非互動示範：以腳本化回答跑完整問卷流程，寫入本地 CSV，並展示注入防護。

本 demo 不使用 input()，而是把一位受訪者的逐題回答以「腳本」餵入既有對話流程
（run_survey → review_and_confirm → save_record），因此走的是與真人作答完全相同的程式路徑。

重點展示：
  - 至少三題以「自由文字」作答，交由 common 層（預設 Groq llama-3.3-70b）解析歸類到合法選項；
  - 一題（改善建議）故意輸入以「=」開頭的惡意公式，驗證「公式/CSV 注入」防護（前導單引號）；
  - 無 Google service account 憑證 → 資料 sink 自動降級為本地 CSV（responses_local.csv）；
  - 跑完印出實際寫入的每一欄，以及注入防護「前 / 後」的值對照。

執行：
    python run_demo.py

需要環境變數：GROQ_API_KEY（LLM，預設供應商）。可用 .env 提供；本 demo 不需 Google 憑證。
（LLM 呼叫次數＝選擇題數量 5 次，temperature=0、max_tokens 小，成本可控。）
"""

from __future__ import annotations

import os
import sys

# 讓本檔可從任意工作目錄執行：切到自身所在目錄，使 responses_local.csv /
# pending_submissions/ 都落在 part2/03_survey_to_google_sheet/ 下。
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
sys.path.insert(0, _HERE)

import survey_agent as sa  # noqa: E402

# 選用：若有 .env 則載入（GROQ_API_KEY / MISTRAL_API_KEY 等）。
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# 強制走「本地 CSV」降級路徑：確保 demo 不需 Google 憑證即可完整跑完並可重現。
os.environ.pop("GOOGLE_SERVICE_ACCOUNT_FILE", None)


# 一位受訪者的腳本化逐題回答（順序對應 survey_agent.QUESTIONS）。
# 註記哪些是「自由文字 → 交給 LLM 歸類」，哪一題是注入攻擊測試。
SCRIPT = [
    ("name",         "Ada Lovelace"),                              # 開放題：原樣保留、不進 LLM
    ("role",         "我平常主要是寫程式的工程師"),                 # 自由文字 → LLM 應歸類為「開發者」
    ("frequency",    "差不多每天都會打開來用"),                     # 自由文字 → LLM 應歸類為「每天」
    ("features",     "最常看儀表板，偶爾也會串接 API"),             # 自由文字 → LLM 應歸類為「儀表板；API」
    ("satisfaction", "算是蠻滿意的，給 4 分吧"),                    # 自由文字 → LLM 應歸類為「4」
    ("improvement",  '=HYPERLINK("https://evil.example/steal","點我領獎")'),  # 惡意公式注入測試（開放題）
    ("interview",    "可以啊沒問題"),                               # 自由文字 → LLM 應歸類為「願意」
]


class ScriptedResponder:
    """把腳本化回答依序餵給 prompt()；用盡後回空字串（於複查階段即代表『送出』）。

    同時把「題目提示 + 回答」印出來，讓非互動流程仍可觀察逐題問答過程。
    """

    def __init__(self, replies):
        self._it = iter(replies)

    def __call__(self, label: str) -> str:
        try:
            reply = next(self._it)
        except StopIteration:
            reply = ""  # 複查時的 prompt 收到空字串 → 直接送出
        print(f"{label}{reply}")
        return reply.strip()


def main() -> None:
    # 實際供應商/模型由環境變數決定，動態顯示以免與真正使用的 provider 不符
    _llm_provider = os.getenv("LLM_PROVIDER", "groq")
    _llm_model = os.getenv("LLM_MODEL", "(該 provider 預設模型)")
    print("=" * 72)
    print(f"非互動 Demo：對話式問卷 → 本地 CSV（common 層 LLM = {_llm_provider} / {_llm_model}）")
    print("=" * 72)

    responder = ScriptedResponder([r for _, r in SCRIPT])

    # do_resend=False：demo 專注在本次作答，不去動先前積壓的待補檔。
    outcome = sa.run_agent(input_fn=responder, do_resend=False)

    if not outcome:
        print("\nDemo 未完成（可能未設定 GROQ_API_KEY，或流程被中止）。")
        print("請設定環境變數 GROQ_API_KEY 後重試：python run_demo.py")
        return

    record = outcome["record"]
    result = outcome["result"]

    # (1) 實際寫入的每一欄（已套用注入防護、多選已攤平）。
    row = sa._build_row(record)
    print("\n" + "-" * 72)
    print("實際寫入資料 sink 的一列（欄位 → 值）：")
    for col, val in zip(sa.HEADER, row):
        print(f"  {col:<15} = {val!r}")
    print(f"\nsink 結果：{result['sink']} —— {result['detail']}")

    # (2) 注入防護「前 / 後」對照（以惡意的 improvement 欄為例）。
    raw_improve = record["answers"].get("improvement", "")
    safe_improve = sa._sanitize_cell(raw_improve)
    print("\n" + "-" * 72)
    print("公式/CSV 注入防護示範（improvement 欄）：")
    print(f"  防護前（受訪者原始輸入）：{raw_improve!r}")
    print(f"  防護後（實際寫入值）    ：{safe_improve!r}")
    print(f"  → 已加前導單引號、且以純文字寫入，Sheets/CSV 匯出後不會被當公式執行："
          f"{'是' if safe_improve.startswith(chr(39)) else '否'}")

    # (3) 開放題保留原文（name 欄未被 LLM 改寫）。
    print("\n" + "-" * 72)
    print("驗證要點：")
    print(f"  - 開放題 name 原樣保留：{record['answers'].get('name')!r}")
    print(f"  - 自由文字經 LLM 歸類：role={record['answers'].get('role')!r}, "
          f"frequency={record['answers'].get('frequency')!r}, "
          f"features={record['answers'].get('features')!r}, "
          f"satisfaction={record['answers'].get('satisfaction')!r}, "
          f"interview={record['answers'].get('interview')!r}")
    print(f"  - 本地 CSV 檔：{os.path.abspath(sa.LOCAL_CSV)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
