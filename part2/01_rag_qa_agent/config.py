"""
全域設定：進階 RAG 調參、.env 載入。

把「會變動的設定」集中一處，rag.py / agent.py / main.py / run_demo.py 皆由此匯入，
避免門檻與調參在多個檔案之間重複、失同步。

模型與供應商已抽離到共用 provider 層（part2/common）：
  - LLM      走 OpenAI 相容端點，預設 Groq `llama-3.3-70b-versatile`
  - Embedding 走 Mistral `mistral-embed`（1024 維、多語含繁中）
兩者皆以環境變數切換（見下方說明與 README），本檔不再寫死模型 ID。
"""
from __future__ import annotations

import pathlib

try:
    # python-dotenv 為選用相依：有裝就自動載入專案的 .env，讓共用層讀得到
    #   GROQ_API_KEY / MISTRAL_API_KEY 等金鑰（金鑰走環境變數，不寫死於程式）。
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # 未安裝 dotenv 也不影響——只要金鑰已由 shell / 系統環境變數提供即可。
    pass

# ------------------------------------------------------------------ #
# 供應商 / 模型：由共用層 part2/common 透過環境變數選擇（此處僅列出預設，供參考）
#   LLM_PROVIDER   預設 groq        LLM_MODEL   預設 llama-3.3-70b-versatile
#   EMBED_PROVIDER 預設 mistral     EMBED_MODEL 預設 mistral-embed
# 金鑰：GROQ_API_KEY（LLM）、MISTRAL_API_KEY（Embedding）。
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# 檢索管線調參
# 註：示範文件庫僅約數個切塊，RETRIEVE_TOP_K / RERANK_TOP_N 在此規模形同虛設，
#     無法充分體現「粗召回 → 重排」價值；正式環境請依文件量調整並擴充知識庫。
# ------------------------------------------------------------------ #
REWRITE_QUERIES = True     # 是否啟用多查詢改寫（無對話脈絡的自足問題會自動略過改寫以省成本）
RETRIEVE_TOP_K = 8         # 向量粗召回數（以 mistral-embed cosine 取 top-k）
RERANK_TOP_N = 4           # 重排後保留數
# 重排相關度門檻(0,1)：改用 LLM 對候選段落打「相關度分數」，1=直接回答、0=完全無關。
# 低於此值視為不相關並剔除，讓「查無規定」的誠實拒答路徑可被觸發。須以驗證集校準。
RERANK_MIN_SCORE = 0.50
CHUNK_SIZE = 480           # 每塊約略字元
CHUNK_OVERLAP = 80

# ------------------------------------------------------------------ #
# 代理迴圈 / 回覆長度
# ------------------------------------------------------------------ #
MAX_TOOL_ROUNDS = 6        # 代理迴圈工具往返上限，避免模型無限要求工具而失控燒錢
ANSWER_MAX_TOKENS = 1024   # 主回答 / 工具迴圈每回合的輸出上限
REWRITE_MAX_TOKENS = 300   # 查詢改寫輸出上限（僅需一小段 JSON 陣列）
RERANK_MAX_TOKENS = 400    # LLM 重排評分輸出上限（僅需一小段 JSON）

# ------------------------------------------------------------------ #
# 範例文件資料夾（相對於本檔；讓程式無需外部資料即可跑起來）
# ------------------------------------------------------------------ #
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
