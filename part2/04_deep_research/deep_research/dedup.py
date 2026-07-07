# -*- coding: utf-8 -*-
"""近似重複偵測與分群（純本地計算，不佔 LLM 預算）。

修正對照：
  * 2.4-44 以字元 n-gram 取代空白斷詞，讓 CJK（中日韓、無空白）也能有效產生特徵。
  * 2.4-46 過短內容（特徵過少）直接跳過去重登記；並以 MinHash+LSH 分桶取候選，
           避免 O(n^2) 全比對。
  * 2.4-15 提供 verify 所需的最小可行分群實作（本地 Jaccard 凝聚式，不呼叫 API）。

近似重複採 MinHash + LSH 取候選、再以精確 Jaccard 覆核（門檻可調），對 CJK 內容比
字元 SimHash 的漢明距離更穩定（後者對短/中文文本雜訊大，微幅改動即翻動多個位元）。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List

from .config import (
    DEDUP_MIN_FEATURES,
    NGRAM_N,
    DEDUP_JACCARD_THRESHOLD,
    CLUSTER_JACCARD_THRESHOLD,
)

_WS_RE = re.compile(r"\s+")
_MASK64 = (1 << 64) - 1
_MERSENNE = (1 << 61) - 1

# MinHash 參數：NUM_HASHES = BANDS * ROWS。b=16, r=4 → LSH 門檻約 0.5，
# 對 Jaccard ≥ 0.5 的配對有高機率同桶，再由精確 Jaccard 覆核到目標門檻。
NUM_HASHES = 64
BANDS = 16
ROWS = 4

# 以固定係數產生 NUM_HASHES 組雜湊（a*x+b mod 大質數），確保跨程序決定性。
_COEFFS = [((i * 2654435761 + 40503) % _MERSENNE or 1,
            (i * 2246822519 + 3266489917) % _MERSENNE) for i in range(1, NUM_HASHES + 1)]


def char_ngrams(text: str, n: int = NGRAM_N) -> set:
    """壓縮空白後取字元 n-gram，對 CJK 與拉丁文皆有效。"""
    t = _WS_RE.sub("", (text or "").lower())
    if not t:
        return set()
    if len(t) <= n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _hash64(s: str) -> int:
    """穩定的 64-bit 雜湊（FNV-1a），不受 PYTHONHASHSEED 影響。"""
    h = 0xCBF29CE484222325
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x100000001B3) & _MASK64
    return h


def _minhash(features: set) -> tuple:
    hashed = [_hash64(f) % _MERSENNE for f in features]
    sig = []
    for a, b in _COEFFS:
        sig.append(min((a * h + b) % _MERSENNE for h in hashed))
    return tuple(sig)


class ContentDeduper:
    """MinHash + LSH 近似重複偵測；候選再以精確 Jaccard 覆核。"""

    def __init__(self, threshold: float = DEDUP_JACCARD_THRESHOLD):
        self.threshold = threshold
        self._features: List[set] = []
        self._buckets = defaultdict(list)  # (band_index, band_tuple) -> [record_index]

    def _band_keys(self, sig: tuple):
        for bi in range(BANDS):
            yield (bi, sig[bi * ROWS : (bi + 1) * ROWS])

    def is_duplicate(self, text: str) -> bool:
        """若為近似重複回傳 True；否則登記後回傳 False。

        過短內容（特徵不足）不登記且視為非重複（2.4-46），交由上游其他鍵去重。
        """
        feats = char_ngrams(text)
        if len(feats) < DEDUP_MIN_FEATURES:
            return False
        sig = _minhash(feats)
        keys = list(self._band_keys(sig))
        candidates = set()
        for key in keys:
            candidates.update(self._buckets.get(key, ()))  # 只比對同桶候選
        for idx in candidates:
            if jaccard(feats, self._features[idx]) >= self.threshold:
                return True
        idx = len(self._features)
        self._features.append(feats)
        for key in keys:
            self._buckets[key].append(idx)
        return False


def cluster_texts(texts: List[str], threshold: float = CLUSTER_JACCARD_THRESHOLD) -> List[List[int]]:
    """以 union-find 對文本做本地凝聚式分群，回傳索引群組列表（2.4-15）。"""
    n = len(texts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    feats = [char_ngrams(t) for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(feats[i], feats[j]) >= threshold:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())
