"""嵌入向量：改走 `part2/common/` 的 embed_texts（預設 Mistral mistral-embed，1024 維）。

設計要點：
    - 語意向量統一由 common.embed_texts 走 API 取得，不需本地嵌入模型（common 內建自動
      分批 + 退避重試，對應 2.2-24 / 2.2-30 的分批與重試精神）。
    - 保留 matrix()：把有 embedding 的 chunk 疊成「正規化矩陣」，供對齊層以內積算 cosine，
      並在空/全缺向量時早退（對應 2.2-4）。
    - 保留一個「確定性字元 n-gram 雜湊」後端作為離線降級：當 EMBEDDING_BACKEND=hash，或
      EMBED provider 金鑰缺失時自動退回（明確警示），讓 demo 不因缺 key 而中斷。此後端為
      **非語意**近似，僅供結構性測試，不代表語意對齊品質。
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Callable, List

import numpy as np

from . import config
from .models import Chunk

# 由 common 抽象層取得 provider 無關的 embeddings（預設 Mistral）。
from common import embed_texts as _common_embed_texts
from common import EMBED_PROVIDERS

log = logging.getLogger(__name__)

# 實際嵌入函式快取（lazy 初始化）。
_embed_fn: Callable[[List[str]], List[List[float]]] | None = None


def _embed_provider_key_present() -> bool:
    """檢查目前 EMBED_PROVIDER 對應的金鑰是否存在（決定要不要退回離線 hash 後端）。"""
    provider = os.getenv("EMBED_PROVIDER", "mistral").strip().lower()
    spec = EMBED_PROVIDERS.get(provider)
    if not spec:
        return False
    _, key_env, _ = spec
    return bool(os.getenv(key_env))


def _api_embedder() -> Callable[[List[str]], List[List[float]]]:
    """走 common.embed_texts（Mistral 等 OpenAI 相容 embeddings；含自動分批與重試）。"""

    def _fn(texts: List[str]) -> List[List[float]]:
        return _common_embed_texts(texts)

    return _fn


def _hash_embedder() -> Callable[[List[str]], List[List[float]]]:
    """確定性雜湊向量（字元 unigram + bigram 詞袋）：離線、無相依、非語意。

    去除所有空白後對每個字元與相鄰字元對雜湊到固定維度，讓「無空白的中文」也能產生
    可區分的向量（純以 whitespace 切詞對中文幾乎沒有訊號）。僅供離線結構性測試。
    """
    dim = config.HASH_EMBED_DIM

    def _feats(t: str):
        s = "".join(t.split())
        yield from s
        for i in range(len(s) - 1):
            yield s[i : i + 2]

    def _fn(texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for tok in _feats(t):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
            out.append(vec.tolist())
        return out

    return _fn


def _get_embedder() -> Callable[[List[str]], List[List[float]]]:
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn

    backend = config.EMBEDDING_BACKEND
    if backend == "hash":
        _embed_fn = _hash_embedder()
        log.warning("嵌入後端：hash（確定性雜湊，非語意，僅供離線結構性測試）")
    elif backend == "api":
        if _embed_provider_key_present():
            _embed_fn = _api_embedder()
            log.info(
                "嵌入後端：api（common.embed_texts，EMBED_PROVIDER=%s）",
                os.getenv("EMBED_PROVIDER", "mistral"),
            )
        else:
            provider = os.getenv("EMBED_PROVIDER", "mistral")
            key_env = EMBED_PROVIDERS.get(provider, ("", "?", ""))[1]
            log.warning(
                "缺少 %s（EMBED_PROVIDER=%s）→ 自動退回離線 hash 嵌入（非語意）。"
                "如需語意對齊品質，請設定該金鑰以使用 %s embeddings。",
                key_env, provider, provider,
            )
            _embed_fn = _hash_embedder()
    else:
        raise ValueError(f"未知的 EMBEDDING_BACKEND：{backend!r}（可選 api / hash）")
    return _embed_fn


def embed_chunks(chunks: List[Chunk]) -> None:
    """就地為每個 chunk 掛上 embedding。

    超長 chunk 先截斷（MAX_CHUNK_CHARS）再嵌入；分批與退避重試由 common.embed_texts
    內建處理（hash 後端則為純本地計算）。
    """
    if not chunks:
        return
    embedder = _get_embedder()
    texts = [c.text[: config.MAX_CHUNK_CHARS] for c in chunks]
    embs = embedder(texts)
    for c, e in zip(chunks, embs):
        c.embedding = list(e)


def matrix(chunks: List[Chunk]):
    """把「有 embedding」的 chunk 疊成正規化矩陣，回傳 (有效清單, 矩陣)。

    空清單或全數缺 embedding 時早退（對應 2.2-4），避免對 shape=(0,) 的 1 維陣列以
    axis=1 呼叫 np.linalg.norm 而拋 AxisError；同時濾掉未成功嵌入（embedding=None）的 chunk。
    """
    valid = [c for c in chunks if c.embedding is not None]
    if not valid:
        return [], np.empty((0, 0), dtype=np.float32)
    m = np.asarray([c.embedding for c in valid], dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return valid, m / (norms + 1e-9)  # 正規化 → 內積即 cosine
