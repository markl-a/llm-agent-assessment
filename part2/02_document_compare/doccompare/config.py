"""集中管理的可調常數（皆可由環境變數覆寫）。

所有強影響結果的參數都集中在此並附調參依據，避免散落各處的魔術數字。每個值都可由
環境變數覆寫，方便在不同文件型態、不同速率上限下校準，而不需改動程式碼。

供應商切換：本專案的 LLM / Embedding 呼叫皆走 `part2/common/` 的共用抽象層
（OpenAI 相容）。供應商與模型由 common 讀取環境變數決定：
    LLM_PROVIDER   (預設 groq)          / LLM_MODEL   (預設 llama-3.3-70b-versatile)
    EMBED_PROVIDER (預設 mistral)        / EMBED_MODEL (預設 mistral-embed，1024 維)
金鑰各自來自 GROQ_API_KEY / MISTRAL_API_KEY，供應商與模型設定一律由 common 統一管理。

註：字元/token 的粗估對「中文」會低估——中文每字約 ≥1 token；因此「切塊尺寸」僅用
字元做粗估，而「合併/切分是否超預算」等關鍵決策改用保守的字元上界估算（見 reduce_stage）。
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# 把 part2/ 加入匯入路徑，讓各模組能 `from common import ...`。
# 本檔位於 .../part2/02_document_compare/doccompare/config.py，往上三層即 part2。
# ---------------------------------------------------------------------------
_PART2_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PART2_DIR not in sys.path:
    sys.path.insert(0, _PART2_DIR)

try:
    # 讓 .env 中的設定自動載入；未安裝 python-dotenv 時安靜略過。
    from dotenv import load_dotenv

    # 優先載入本專案資料夾下的 .env，其次沿用預設搜尋。
    _PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_PROJ_DIR, ".env"))
    load_dotenv()
except Exception:  # pragma: no cover - dotenv 為選配
    pass


def _get_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# 嵌入後端
#   api （預設）：走 common.embed_texts（預設 Mistral mistral-embed，1024 維，多語含繁中）。
#   hash        ：確定性字元 n-gram 雜湊向量，無需任何 key／下載，**非語意**，僅供離線
#                 結構性測試；當 EMBED provider 金鑰缺失時，embeddings 也會自動退回此模式，
#                 讓 demo 不致因缺 key 而中斷（會明確警示）。
# ---------------------------------------------------------------------------
EMBEDDING_BACKEND: str = _get_str("EMBEDDING_BACKEND", "api").strip().lower()
HASH_EMBED_DIM: int = _get_int("HASH_EMBED_DIM", 512)

# ---------------------------------------------------------------------------
# 切塊參數（僅粗估；關鍵尺寸決策改用保守字元上界估算校準）
# ---------------------------------------------------------------------------
TARGET_CHARS: int = _get_int("TARGET_CHARS", 1600)   # 每塊目標字元
OVERLAP_CHARS: int = _get_int("OVERLAP_CHARS", 200)  # 重疊字元，維持跨塊語意連續
MAX_CHUNK_CHARS: int = _get_int("MAX_CHUNK_CHARS", 8000)  # 超長 chunk 嵌入前先截斷

# ---------------------------------------------------------------------------
# 對齊參數（門檻應以標註樣本校準，非魔術數字）
# ---------------------------------------------------------------------------
ALIGN_THRESHOLD: float = _get_float("ALIGN_THRESHOLD", 0.72)
ALIGN_BLOCK: int = _get_int("ALIGN_BLOCK", 1024)      # block-wise 分塊寬度
ALIGN_TOPK: int = _get_int("ALIGN_TOPK", 10)          # 每個 A 段取的候選數（top-k）
USE_FAISS: bool = _get_bool("USE_FAISS", False)       # 大規模時改走 FAISS top-k

# ---------------------------------------------------------------------------
# map / reduce 併發與 token 預算
# ---------------------------------------------------------------------------
MAP_CONCURRENCY: int = _get_int("MAP_CONCURRENCY", 6)
REDUCE_CONCURRENCY: int = _get_int("REDUCE_CONCURRENCY", 3)
MAP_MAX_TOKENS: int = _get_int("MAP_MAX_TOKENS", 1024)
REDUCE_MAX_TOKENS: int = _get_int("REDUCE_MAX_TOKENS", 2048)
REDUCE_TOKEN_BUDGET: int = _get_int("REDUCE_TOKEN_BUDGET", 24_000)  # 單次 reduce 輸入上限（估算字元）
REPORT_MAX_TOKENS_CAP: int = _get_int("REPORT_MAX_TOKENS_CAP", 3000)

# ---------------------------------------------------------------------------
# 速率限制（token-bucket，TPM/RPM）。對 LLM 端做輕量節流，避免瞬間扇出觸發供應商 429；
# 真正的供應商限額仍由 common 內建的退避重試兜底。預設值貼近 Groq 免費層，請依帳號調整。
# ---------------------------------------------------------------------------
LLM_RPM: int = _get_int("LLM_RPM", 28)
LLM_TPM: int = _get_int("LLM_TPM", 12_000)

# ---------------------------------------------------------------------------
# Checkpoint（持久化 map/reduce 中間結果以支援續跑）
# ---------------------------------------------------------------------------
CHECKPOINT_DIR: str = _get_str("CHECKPOINT_DIR", ".doccompare_checkpoints")
