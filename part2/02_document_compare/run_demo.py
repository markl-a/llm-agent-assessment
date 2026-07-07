"""非互動示範：讀兩份範例合約，跑完整 map → align → 逐段差異 → reduce → 報告。

特色：
    - 完全非互動（不使用 input()），可直接 `python run_demo.py` 跑完並印出結構化比較報告。
    - LLM / Embedding 皆走 part2/common/ 共用抽象層（預設 Groq LLM + Mistral Embedding）。
    - 為控制 LLM 呼叫次數與 token：採較小的切塊尺寸讓章節對齊有意義，同時把並發與速率
      控制在保守範圍；範例合約很短，整趟通常只需十餘次 LLM 呼叫。

環境變數（金鑰一律走環境變數，不寫死）：
    GROQ_API_KEY     ── LLM（Groq llama-3.3-70b-versatile，預設）
    MISTRAL_API_KEY  ── Embedding（Mistral mistral-embed，1024 維，預設）
    缺 MISTRAL_API_KEY 時，嵌入會自動退回離線 hash（非語意）以維持可執行，並在日誌警示。
    亦可用 LLM_PROVIDER / LLM_MODEL / EMBED_PROVIDER / EMBED_MODEL 覆寫供應商與模型。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# 把 part2/ 加入匯入路徑，讓 `from common import ...` 可用（與各模組一致）。
sys.path.insert(0, os.path.dirname(_HERE))

# 在匯入 doccompare.config 之前設定 demo 友善的預設（使用者既有環境變數優先）。
# 範例合約每章約 100–325 字，用較小的切塊尺寸讓「章節/條文對齊」有意義又不致產生過多 LLM 呼叫。
os.environ.setdefault("TARGET_CHARS", "220")
os.environ.setdefault("OVERLAP_CHARS", "40")
os.environ.setdefault("ALIGN_THRESHOLD", "0.62")
os.environ.setdefault("MAP_CONCURRENCY", "3")
os.environ.setdefault("REDUCE_CONCURRENCY", "2")

# 報告含 emoji（🔴🟡🟢）與 ≈ 等非 cp950 字元；把主控台輸出改為 UTF-8（失敗則以 replace 兜底），
# 避免在 Windows 傳統 cp950 主控台印出時拋 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:  # 讓本專案資料夾下的 .env 生效（若有）
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_HERE, ".env"))
except Exception:
    pass

from doccompare import llm  # noqa: E402  （放在 sys.path/env 設定之後）
from doccompare.pipeline import compare  # noqa: E402

_A = os.path.join(_HERE, "data", "contract_v1.md")
_B = os.path.join(_HERE, "data", "contract_v2.md")


def _banner() -> None:
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    embed_provider = os.getenv("EMBED_PROVIDER", "mistral")
    embed_model = os.getenv("EMBED_MODEL", "mistral-embed")
    embed_key = bool(os.getenv("MISTRAL_API_KEY")) if embed_provider == "mistral" else True
    print("=" * 72)
    print("兩份大型文件比較 Agent — 非互動示範 (run_demo)")
    print(f"  LLM      : {provider} / {model}")
    if embed_key:
        print(f"  Embedding: {embed_provider} / {embed_model}")
    else:
        print(f"  Embedding: {embed_provider} / {embed_model}"
              "  ← 未設 MISTRAL_API_KEY，將自動退回離線 hash（非語意）")
    print(f"  文件 A   : {os.path.relpath(_A, _HERE)}")
    print(f"  文件 B   : {os.path.relpath(_B, _HERE)}")
    print("=" * 72)


async def _main() -> int:
    _banner()
    if not os.getenv("GROQ_API_KEY") and os.getenv("LLM_PROVIDER", "groq") == "groq":
        print("錯誤：未設定 GROQ_API_KEY，無法呼叫 LLM。請先 `set GROQ_API_KEY=...`。",
              file=sys.stderr)
        return 2
    for p in (_A, _B):
        if not os.path.exists(p):
            print(f"錯誤：找不到範例合約 {p}", file=sys.stderr)
            return 2

    t0 = time.time()
    try:
        report = await compare(_A, _B)
    except Exception as e:  # 給可讀錯誤而非裸 traceback
        print(f"執行失敗：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0

    print()
    print(report)
    print()
    s = llm.stats()
    print("-" * 72)
    print(f"完成：耗時 {elapsed:.1f}s；LLM 呼叫 {s['calls']} 次、"
          f"輸入 token 估算 ≈ {s['input_token_estimate']}")
    print("-" * 72)
    return 0


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "WARNING").upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
