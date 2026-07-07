# -*- coding: utf-8 -*-
"""集中管理的常數與設定。

LLM / Embedding 走共用 provider 抽象層（part2/common）：LLM 走 OpenAI 相容端點
（預設 Groq llama-3.3-70b-versatile），Embedding 走 Mistral（mistral-embed）。
供應商 / 模型一律由環境變數決定（LLM_PROVIDER / LLM_MODEL / EMBED_PROVIDER /
EMBED_MODEL），金鑰走環境變數（GROQ_API_KEY / MISTRAL_API_KEY…）。

模型分層由環境變數控制：如需不同階段用不同模型，可用 LLM_MODEL 覆寫
（例如 reader 換較小模型）。
"""

# ---- LLM 呼叫（走 common 層；供應商/模型由環境變數決定）--------------------
# 各階段的 max_tokens 分層以控制成本 / 時間（規劃/驗證/彙整較長，逐來源閱讀較短）。
PLAN_MAX_TOKENS = 1024
STEP_MAX_TOKENS = 1024              # searcher 每一步（tool-calling）的回覆上限
READER_MAX_TOKENS = 1024           # 逐來源抽取 claim：量大、屬輕量任務
VERIFY_MAX_TOKENS = 900
GAP_MAX_TOKENS = 700
SYNTH_MAX_TOKENS = 4096            # 報告階段

# ---- 抓取 / 引用可驗證範圍 --------------------------------------------------
MAX_SOURCE_CHARS = 16000           # raw_text 可驗證範圍
FETCH_TIMEOUT = 12                 # 秒
FETCH_MAX_BYTES = 3_000_000        # 回應大小上限（SSRF / 資源保護）
FETCH_MAX_REDIRECTS = 5

# ---- Quote-grounding / 引用 ------------------------------------------------
QUOTE_MATCH_THRESHOLD = 0.85       # 整段引文最長公共子串覆蓋率門檻

# ---- 獨立來源 / 信心 -------------------------------------------------------
MIN_INDEPENDENT_SOURCES = 2        # 至少 2 個不同（正規化後）網域
CONFIDENCE_SATURATION = 3          # 獨立來源數飽和上限
HIGH_CONFIDENCE = 0.6              # 視為「已覆蓋」的信心門檻

# ---- 迴圈 / 預算 -----------------------------------------------------------
MAX_ITERATIONS = 3                 # 收斂主迴圈最大迭代（run_demo 會下修為 1）
MAX_SEARCHER_STEPS = 6             # 單一子問題 searcher 工具迴圈上限（tool_loop max_rounds）
MAX_RESULTS_PER_SEARCH = 6
MAX_SOURCES_PER_SUBQ = 5
DEFAULT_TOKEN_BUDGET = 600_000     # 加權 token 預算（見 models.Usage.billed；common 層不回傳
                                   # 用量，改以字元數估算，仍足以驅動「預算耗盡即收斂」控制）

# ---- Embedding 輔助（走 Mistral API）--------------------------------------
# 以向量相似度對搜尋結果做輕量 rerank（取代本地 cross-encoder）；缺 MISTRAL_API_KEY
# 或任何錯誤都會自動降級為原順序，不阻斷主流程。
USE_EMBED_RERANK = True

# ---- 去重 ------------------------------------------------------------------
DEDUP_JACCARD_THRESHOLD = 0.8      # 近似重複的精確 Jaccard 覆核門檻（MinHash+LSH 取候選）
DEDUP_MIN_FEATURES = 8             # 特徵過少（過短內容）直接跳過登記
NGRAM_N = 3                        # CJK 友善的字元 n-gram
CLUSTER_JACCARD_THRESHOLD = 0.5    # verify 分群門檻（純本地計算，不佔預算）

USER_AGENT = "DeepResearchAgent/1.0 (+https://example.invalid/bot)"
