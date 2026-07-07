"""章節/段落對齊配對。

修正對應：
    2.2-2   不物化 O(n×m) 稠密相似度矩陣：預設 block-wise 分塊（僅持有 BLOCK×m），
            大規模改走 FAISS/HNSW 取 top-k 候選（USE_FAISS=1 且已安裝 faiss）。
    2.2-4   空/掃描文件早退，不對空陣列做 axis=1 運算。
    2.2-5   以全體 (a,b,score) 由高到低排序後指派 1:1（非依 A 索引順序的貪婪 argmax）；
            最佳 B 被較高分的 A 佔用時，A 自然回退次佳未使用且過門檻的 B。
    2.2-13  先以 section/heading 粗分桶（章節化文件最強的對齊訊號），再於同章節內做
            embedding 細對齊，跨章節殘餘再做一次全域對齊——section metadata 真正進入
            對齊邏輯，符合架構圖描述。

回傳 pairs：List[(a, b, score)]，其中 b=None 表示 A 獨有、a=None 表示 B 獨有。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from . import config
from .embeddings import matrix
from .models import Chunk

log = logging.getLogger(__name__)

Pair = Tuple[Optional[Chunk], Optional[Chunk], float]


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def _candidates_blockwise(
    a_list: List[Chunk],
    b_list: List[Chunk],
    rowmap: Dict[int, np.ndarray],
    threshold: float,
    block: int,
) -> Iterator[Tuple[float, Chunk, Chunk]]:
    """block-wise 產生過門檻的 (score, a, b)。

    每次只計算 A[s:s+block] @ B.T（BLOCK×m），即時套門檻後丟棄，**不物化整個 n×m**。
    """
    if not a_list or not b_list:
        return
    A = np.asarray([rowmap[id(c)] for c in a_list], dtype=np.float32)
    B = np.asarray([rowmap[id(c)] for c in b_list], dtype=np.float32)
    for s in range(0, len(a_list), block):
        sim = A[s : s + block] @ B.T  # BLOCK×m，非 n×m
        for r, c in np.argwhere(sim >= threshold):
            yield float(sim[r, c]), a_list[s + int(r)], b_list[int(c)]


def _candidates_faiss(
    a_list: List[Chunk],
    b_list: List[Chunk],
    rowmap: Dict[int, np.ndarray],
    threshold: float,
    topk: int,
) -> Iterator[Tuple[float, Chunk, Chunk]]:
    """FAISS top-k 候選：向量已正規化，內積索引即 cosine；永不物化 n×m。"""
    import faiss

    if not a_list or not b_list:
        return
    A = np.ascontiguousarray([rowmap[id(c)] for c in a_list], dtype=np.float32)
    B = np.ascontiguousarray([rowmap[id(c)] for c in b_list], dtype=np.float32)
    index = faiss.IndexFlatIP(B.shape[1])
    index.add(B)
    k = min(topk, len(b_list))
    scores, idxs = index.search(A, k)
    for i in range(len(a_list)):
        for score, j in zip(scores[i], idxs[i]):
            if j < 0:
                continue
            if score >= threshold:
                yield float(score), a_list[i], b_list[int(j)]


def _candidates(
    a_list: List[Chunk],
    b_list: List[Chunk],
    rowmap: Dict[int, np.ndarray],
    threshold: float,
) -> Iterator[Tuple[float, Chunk, Chunk]]:
    if config.USE_FAISS and _faiss_available():
        return _candidates_faiss(a_list, b_list, rowmap, threshold, config.ALIGN_TOPK)
    return _candidates_blockwise(a_list, b_list, rowmap, threshold, config.ALIGN_BLOCK)


def _assign(
    cands: Iterator[Tuple[float, Chunk, Chunk]],
    pairs: List[Pair],
    matched_a: set,
    matched_b: set,
) -> None:
    """以全體 (a,b,score) 由高到低排序後指派 1:1（避免貪婪誤判）。"""
    for score, a, b in sorted(cands, key=lambda t: t[0], reverse=True):
        if id(a) in matched_a or id(b) in matched_b:
            continue
        matched_a.add(id(a))
        matched_b.add(id(b))
        pairs.append((a, b, score))


def align(
    chunks_a: List[Chunk],
    chunks_b: List[Chunk],
    threshold: float = config.ALIGN_THRESHOLD,
) -> List[Pair]:
    """先章節內細對齊，再跨章節殘餘全域對齊，最後標各自獨有。"""
    if not chunks_a and not chunks_b:
        return []

    va, ma = matrix(chunks_a)
    vb, mb = matrix(chunks_b)
    if chunks_a and not va:
        raise ValueError("A 文件無有效 embedding（疑為掃描/空白 PDF，請先 OCR）")
    if chunks_b and not vb:
        raise ValueError("B 文件無有效 embedding（疑為掃描/空白 PDF，請先 OCR）")

    rowmap: Dict[int, np.ndarray] = {}
    rowmap.update({id(c): ma[i] for i, c in enumerate(va)})
    rowmap.update({id(c): mb[i] for i, c in enumerate(vb)})

    pairs: List[Pair] = []
    matched_a: set = set()
    matched_b: set = set()

    # 1) 章節內細對齊（section metadata 進入對齊邏輯）
    ba: Dict[Optional[str], List[Chunk]] = defaultdict(list)
    bb: Dict[Optional[str], List[Chunk]] = defaultdict(list)
    for c in va:
        ba[c.section].append(c)
    for c in vb:
        bb[c.section].append(c)
    for sec in set(ba) & set(bb):
        _assign(_candidates(ba[sec], bb[sec], rowmap, threshold),
                pairs, matched_a, matched_b)

    # 2) 跨章節殘餘全域對齊（標題被改名等情形）
    rest_a = [c for c in va if id(c) not in matched_a]
    rest_b = [c for c in vb if id(c) not in matched_b]
    _assign(_candidates(rest_a, rest_b, rowmap, threshold),
            pairs, matched_a, matched_b)

    # 3) 各自獨有
    for c in va:
        if id(c) not in matched_a:
            pairs.append((c, None, 0.0))
    for c in vb:
        if id(c) not in matched_b:
            pairs.append((None, c, 0.0))

    log.info(
        "對齊完成：配對 %d 對、A 獨有 %d、B 獨有 %d",
        sum(1 for a, b, _ in pairs if a is not None and b is not None),
        sum(1 for a, b, _ in pairs if b is None),
        sum(1 for a, b, _ in pairs if a is None),
    )
    return pairs
