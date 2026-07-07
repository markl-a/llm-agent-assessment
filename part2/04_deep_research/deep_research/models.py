# -*- coding: utf-8 -*-
"""型別化的資料模型：Usage / Budget / Source / Claim / VerifiedFinding /
SubQuestion / ResearchState。

ResearchState 為可序列化結構，orchestrator 以 save_checkpoint / load_checkpoint
提供真正的 checkpoint 與重跑（2.4-30）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 用量與預算
# ---------------------------------------------------------------------------
@dataclass
class Usage:
    """單次 API 呼叫的 token 用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_sdk(cls, u: Any) -> "Usage":
        g = lambda name: int(getattr(u, name, 0) or 0)
        return cls(
            input_tokens=g("input_tokens"),
            output_tokens=g("output_tokens"),
            cache_read_input_tokens=g("cache_read_input_tokens"),
            cache_creation_input_tokens=g("cache_creation_input_tokens"),
        )

    @property
    def billed(self) -> float:
        """加權用量。cache_creation≈1.25×、cache_read≈0.1×；thinking 已計入
        output（2.4-28：一旦開啟 prompt caching 也不失真）。"""
        return (
            self.input_tokens
            + self.output_tokens
            + 1.25 * self.cache_creation_input_tokens
            + 0.1 * self.cache_read_input_tokens
        )


@dataclass
class Budget:
    """全域加權 token 預算；add() 以鎖保護，可安全用於平行化。"""

    total: float
    spent: float = 0.0
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, usage: Usage) -> None:
        with self._lock:
            self.spent += usage.billed

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.total

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.spent)


# ---------------------------------------------------------------------------
# 研究實體
# ---------------------------------------------------------------------------
@dataclass
class Source:
    id: int
    url: str
    canonical_url: str
    domain: str
    registered_domain: str
    title: str
    credibility: float
    raw_text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(**d)


@dataclass
class Claim:
    id: int
    subquestion_id: int
    source_id: int
    text: str
    quote: str
    supported: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        return cls(**d)


@dataclass
class VerifiedFinding:
    subquestion_id: int
    statement: str
    supporting: list          # List[int]：支持該陳述的 source id
    conflicting: list          # List[int]：反駁該陳述的 source id（2.4-13）
    confidence: float
    contradicted: bool
    cluster_key: str           # 跨迭代去重 / 更新的鍵（2.4-12）

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VerifiedFinding":
        return cls(**d)


@dataclass
class SubQuestion:
    id: int
    text: str
    covered: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SubQuestion":
        return cls(**d)


@dataclass
class ResearchState:
    question: str
    subquestions: list = field(default_factory=list)      # List[SubQuestion]
    sources: dict = field(default_factory=dict)           # id(int) -> Source
    claims: list = field(default_factory=list)            # List[Claim]
    findings: list = field(default_factory=list)          # List[VerifiedFinding]
    seen_canonical: set = field(default_factory=set)      # 已收錄的 canonical_url
    verify_cache: dict = field(default_factory=dict)      # cluster_key -> finding dict
    iterations: int = 0
    incomplete: bool = False                              # 因預算/覆蓋不足而不完整

    _next_source_id: int = 0
    _next_claim_id: int = 0
    _next_subq_id: int = 0
    # 內容去重器（ContentDeduper）；不直接序列化，載入時由 source 文本重建。
    deduper: Any = field(default=None, repr=False, compare=False)
    _write_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    # ---- id 產生器 --------------------------------------------------------
    def new_source_id(self) -> int:
        sid = self._next_source_id
        self._next_source_id += 1
        return sid

    def new_claim_id(self) -> int:
        cid = self._next_claim_id
        self._next_claim_id += 1
        return cid

    def add_subquestion(self, text: str) -> SubQuestion:
        sq = SubQuestion(id=self._next_subq_id, text=text)
        self._next_subq_id += 1
        self.subquestions.append(sq)
        return sq

    # ---- 序列化（checkpoint）---------------------------------------------
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "subquestions": [s.to_dict() for s in self.subquestions],
            "sources": {str(k): v.to_dict() for k, v in self.sources.items()},
            "claims": [c.to_dict() for c in self.claims],
            "findings": [f.to_dict() for f in self.findings],
            "seen_canonical": sorted(self.seen_canonical),
            "verify_cache": self.verify_cache,
            "iterations": self.iterations,
            "incomplete": self.incomplete,
            "_next_source_id": self._next_source_id,
            "_next_claim_id": self._next_claim_id,
            "_next_subq_id": self._next_subq_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchState":
        st = cls(question=d["question"])
        st.subquestions = [SubQuestion.from_dict(x) for x in d.get("subquestions", [])]
        st.sources = {int(k): Source.from_dict(v) for k, v in d.get("sources", {}).items()}
        st.claims = [Claim.from_dict(x) for x in d.get("claims", [])]
        st.findings = [VerifiedFinding.from_dict(x) for x in d.get("findings", [])]
        st.seen_canonical = set(d.get("seen_canonical", []))
        st.verify_cache = dict(d.get("verify_cache", {}))
        st.iterations = int(d.get("iterations", 0))
        st.incomplete = bool(d.get("incomplete", False))
        st._next_source_id = int(d.get("_next_source_id", len(st.sources)))
        st._next_claim_id = int(d.get("_next_claim_id", len(st.claims)))
        st._next_subq_id = int(d.get("_next_subq_id", len(st.subquestions)))
        return st
