# -*- coding: utf-8 -*-
"""離線黃金集回歸測試（不需 API 金鑰，可納入 CI）。

對 quote_supported、validate_citations、score_domain、canonical_url、去重與信心公式
做回歸，避免調參後靜默退化（對應可觀測性/評估修正 2.4-31）。

執行：
    python -m eval.golden          （於 part2/04_deep_research/ 目錄下）
"""
from __future__ import annotations

import sys

from deep_research.dedup import ContentDeduper, cluster_texts, jaccard, char_ngrams
from deep_research.grounding import quote_supported, validate_citations
from deep_research.net import canonical_url, registered_domain, score_domain
from deep_research.orchestrator import compute_confidence

_failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        _failures.append(name)


def run() -> int:
    # --- quote-grounding（2.4-18）---------------------------------------
    src = "根據世界衛生組織的統計，2023 年全球平均氣溫較工業化前上升約 1.1 攝氏度。"
    check("quote 命中（含空白差異）", quote_supported("2023 年全球平均氣溫較工業化前上升約 1.1 攝氏度", src))
    check("quote 捏造應拒絕", not quote_supported("2023 年全球平均氣溫下降了 5 度並導致冰河期", src))

    # --- 引用驗證（2.4-37）---------------------------------------------
    cleaned, used, invalid = validate_citations("結論 A [1]，結論 B [2][9]。", {1, 2})
    check("越界引用被移除", "[9]" not in cleaned and 9 in invalid and used == {1, 2})

    # --- 網域評分邊界（2.4-8 / 2.4-41）---------------------------------
    check("who.int 高可信", score_domain("who.int") == 0.9)
    check("notarxiv.org 不誤命中 arxiv", score_domain("notarxiv.org") == 0.5)
    check("xxxbbc.com 不誤命中 bbc", score_domain("xxxbbc.com") == 0.5)
    check("nih.gov 由 .gov 涵蓋", score_domain("nih.gov") == 0.9)
    check("blog.nature.com 帶點界線命中", score_domain("blog.nature.com") == 0.9)

    # --- canonical_url（2.4-6）------------------------------------------
    a = canonical_url("https://ex.com/p?id=1&utm_source=x")
    b = canonical_url("https://ex.com/p?id=2&utm_source=y")
    check("不同 query 不被誤判為同一 URL", a != b)
    c1 = canonical_url("https://www.ex.com/p?a=1&b=2&gclid=zz")
    c2 = canonical_url("https://ex.com/p?b=2&a=1")
    check("追蹤參數移除且參數排序後等價", c1 == c2)

    # --- 獨立網域正規化（2.4-16）---------------------------------------
    check("m. 子網域正規化", registered_domain("m.bbc.com") == "bbc.com")
    check("兩段式後綴 eTLD+1", registered_domain("news.gov.tw") == "news.gov.tw" or
          registered_domain("a.b.gov.tw") == "b.gov.tw")

    # --- 去重（2.4-44 / 2.4-46）----------------------------------------
    d = ContentDeduper()
    long_cn = (
        "人工智慧近年在自然語言處理領域取得重大突破，深度學習模型能理解上下文語意，"
        "並於機器翻譯、問答系統、文本摘要等任務展現接近人類的水準。研究者指出，"
        "大型語言模型的規模擴張帶來能力的湧現，但同時也引發能源消耗與偏見等疑慮，"
        "因此可解釋性與對齊研究成為近期學界與產業界共同關注的重點方向。"
    )
    unrelated = "海平面在過去一世紀持續上升，沿海城市面臨淹水與鹽化的威脅，各國正評估防洪工程。"
    check("首次登記非重複", d.is_duplicate(long_cn) is False)
    check("近似重複被偵測", d.is_duplicate(long_cn + "（本文於文末補充一句說明。）") is True)
    check("無關內容非重複", d.is_duplicate(unrelated) is False)
    check("過短內容跳過登記", d.is_duplicate("短") is False)

    # --- 分群（2.4-15）-------------------------------------------------
    groups = cluster_texts(["氣溫上升約 1.1 度", "氣溫上升大約 1.1 度", "海平面持續上升"])
    check("相近主張同群、無關另群", len(groups) == 2)

    # --- 信心公式（2.4-33）---------------------------------------------
    check("單一來源上限 0.4", compute_confidence(1, 0.9, False) <= 0.4)
    check("兩獨立來源可達較高信心", compute_confidence(3, 0.9, False) > 0.6)
    check("矛盾折半", compute_confidence(3, 0.9, True) < compute_confidence(3, 0.9, False))

    print(f"\n{'全部通過' if not _failures else str(len(_failures)) + ' 項失敗'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(run())
