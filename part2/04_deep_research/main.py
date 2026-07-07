# -*- coding: utf-8 -*-
"""Deep Research Agent CLI 進入點。

用法：
    python main.py "你的研究問題" [--budget 600000] [--parallel] [--output report.md]

LLM / Embedding 走共用 provider 抽象層（part2/common）：預設 Groq（LLM）+ Mistral（Embedding）。
搜尋走 DuckDuckGo（免金鑰）。金鑰一律走環境變數 / .env：GROQ_API_KEY、MISTRAL_API_KEY。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# 把 part2/ 加入路徑，讓 deep_research 內部可 `import common`。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_research import (
    Budget,
    DuckDuckGoSearch,
    deep_research,
    load_checkpoint,
)
from deep_research import config


def _load_dotenv() -> None:
    """若安裝了 python-dotenv 就載入 .env；否則靜默略過。"""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep Research Agent（Groq + Mistral，走 common 層）")
    p.add_argument("question", help="研究問題")
    p.add_argument("--budget", type=float, default=config.DEFAULT_TOKEN_BUDGET,
                   help="加權 token 預算（估算值；見 Usage.billed）")
    p.add_argument("--max-subquestions", type=int, default=8, help="規劃階段最多保留幾個子問題")
    p.add_argument("--parallel", action="store_true", help="子問題層平行（asyncio.Semaphore 限流）")
    p.add_argument("--concurrency", type=int, default=4, help="平行併發上限")
    p.add_argument("--allowlist", nargs="*", default=None,
                   help="可選的網域允許清單（eTLD+1），啟用後只抓清單內網域")
    p.add_argument("--checkpoint", default=None, help="checkpoint JSON 路徑（可續跑）")
    p.add_argument("--resume", action="store_true", help="從 --checkpoint 續跑既有研究狀態")
    p.add_argument("--output", default=None, help="報告輸出檔（Markdown）；預設印到 stdout")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    key_env = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
               "cerebras": "CEREBRAS_API_KEY", "together": "TOGETHER_API_KEY",
               "nvidia": "NVIDIA_NIM_API_KEY", "mistral": "MISTRAL_API_KEY"}.get(provider, "GROQ_API_KEY")
    if not os.environ.get(key_env):
        print(f"錯誤：未偵測到 {key_env}（LLM_PROVIDER={provider} 需要它）。"
              f"請設定環境變數或 .env。", file=sys.stderr)
        return 2
    if not os.environ.get("MISTRAL_API_KEY"):
        print("提示：未設定 MISTRAL_API_KEY，搜尋結果的向量 rerank 將自動略過（不影響流程）。",
              file=sys.stderr)

    state = None
    if args.resume and args.checkpoint and os.path.exists(args.checkpoint):
        state = load_checkpoint(args.checkpoint)

    with DuckDuckGoSearch() as search:
        result = deep_research(
            args.question,
            search=search,
            budget=Budget(total=args.budget),
            state=state,
            allowlist=set(args.allowlist) if args.allowlist else None,
            checkpoint_path=args.checkpoint,
            parallel=args.parallel,
            concurrency=args.concurrency,
            max_subquestions=args.max_subquestions,
        )

    report = result["report"]
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"報告已寫入：{args.output}", file=sys.stderr)
    else:
        print(report)

    print("\n---\n指標：", result["metrics"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
