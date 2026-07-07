"""串接整條 pipeline 的 compare() 進入點。

修正對應：
    2.2-26  load()（pdfplumber）與 embed_chunks()（同步 embeddings，走 common.embed_texts）
            皆為阻塞呼叫，以 to_thread 丟到執行緒，避免卡住事件迴圈、令多對文件並發比對互相阻塞。
    2.2-21  持久化 map/reduce 中間結果做 checkpoint，中斷可續跑。

資料流：
    parse → embed → align → (map) → (reduce) → report
dry_run=True 時只跑到 align，輸出結構性報告（不呼叫任何 LLM，可離線 demo）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from . import checkpoint
from .alignment import align
from .embeddings import embed_chunks
from .map_stage import run_map
from .parsing import load
from .reduce_stage import run_reduce
from .report import final_report, structural_report

log = logging.getLogger(__name__)


async def compare(
    path_a: str,
    path_b: str,
    *,
    doc_a_id: str = "A",
    doc_b_id: str = "B",
    dry_run: bool = False,
    resume: bool = False,
) -> str:
    """比較兩份文件，回傳 Markdown 報告字串。"""
    # 解析與嵌入皆為阻塞呼叫 → 丟執行緒，避免卡住事件迴圈（對應 2.2-26）。
    a = await asyncio.to_thread(load, path_a, doc_a_id)
    b = await asyncio.to_thread(load, path_b, doc_b_id)
    log.info("解析完成：A=%d chunks、B=%d chunks", len(a), len(b))

    await asyncio.gather(
        asyncio.to_thread(embed_chunks, a),
        asyncio.to_thread(embed_chunks, b),
    )

    pairs = align(a, b)

    if dry_run:
        # 離線結構性報告，不呼叫 LLM。
        return structural_report(pairs)

    key = checkpoint.run_key(path_a, path_b)

    # ---- MAP（可續跑）----
    cached_map = checkpoint.load(key, "map") if resume else None
    if cached_map is not None:
        results, failures = cached_map["results"], cached_map["failures"]
        log.info("沿用 map checkpoint：成功 %d、失敗 %d", len(results), len(failures))
    else:
        results, failures = await run_map(pairs)
        checkpoint.save(key, "map", {"results": results, "failures": failures})

    # ---- REDUCE（可續跑）----
    cached_reduce = checkpoint.load(key, "reduce") if resume else None
    if cached_reduce is not None:
        sections = cached_reduce
        log.info("沿用 reduce checkpoint：%d 章節", len(sections))
    else:
        sections = await run_reduce(results)
        checkpoint.save(key, "reduce", sections)

    return await final_report(sections, failures)
