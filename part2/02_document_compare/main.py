"""兩份大型文件比較 Agent — CLI 進入點。

LLM / Embedding 皆走 part2/common/ 共用抽象層（預設 Groq LLM + Mistral Embedding）。
需要環境變數：GROQ_API_KEY（LLM）、MISTRAL_API_KEY（Embedding）。缺 MISTRAL_API_KEY 時
嵌入會自動退回離線 hash（非語意）以維持可執行。

用法：
    # 直接跑內建範例（無需真實 PDF；需 GROQ_API_KEY，建議並設 MISTRAL_API_KEY）
    python main.py

    # 指定兩份文件（.pdf / .md / .txt 皆可）
    python main.py --a data/contract_v1.md --b data/contract_v2.md --out report.md

    # 離線結構性 demo（不呼叫任何 LLM，只驗證 解析＋切塊＋對齊）
    #   搭配 EMBEDDING_BACKEND=hash 可完全離線、無需任何 key
    python main.py --dry-run

    # 中斷後續跑（沿用 map/reduce checkpoint）
    python main.py --resume

    # 非互動、印出結構化比較報告的完整示範
    python run_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from doccompare.pipeline import compare

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_A = os.path.join(_HERE, "data", "contract_v1.md")
_DEFAULT_B = os.path.join(_HERE, "data", "contract_v2.md")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="比較兩份大型文件並輸出結構化差異報告（超大 PDF 不塞整份）。"
    )
    p.add_argument("--a", default=_DEFAULT_A, help="文件 A 路徑（.pdf/.md/.txt）")
    p.add_argument("--b", default=_DEFAULT_B, help="文件 B 路徑（.pdf/.md/.txt）")
    p.add_argument("--out", default=None, help="報告輸出檔（預設印到 stdout）")
    p.add_argument("--dry-run", action="store_true", help="只跑到對齊，不呼叫 LLM（可離線）")
    p.add_argument("--resume", action="store_true", help="沿用既有 map/reduce checkpoint 續跑")
    p.add_argument("--verbose", "-v", action="store_true", help="輸出詳細日誌")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> str:
    for path in (args.a, args.b):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到輸入檔：{path}")
    return await compare(
        args.a, args.b, dry_run=args.dry_run, resume=args.resume
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        report = asyncio.run(_run(args))
    except Exception as e:  # 對使用者給出可讀錯誤，而非裸 traceback
        print(f"執行失敗：{e}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"報告已寫入：{args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
