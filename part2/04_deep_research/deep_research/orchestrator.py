# -*- coding: utf-8 -*-
"""收斂主迴圈、信心公式、報告綜整、checkpoint 與可觀測性。

報告彙整走 llm.complete_text（共用 provider 抽象層 part2/common → common.complete）；
供應商 / 模型由環境變數決定。工程手段（信心公式、程式驗證引用、參考文獻生成、覆蓋度附錄、
checkpoint、指標、平行化）為本模組核心。

關鍵設計：
  * compute_confidence 讓獨立來源數成為主因子（需 ≥2 才可能高分），單一來源上限 0.4。
  * synthesize 收到子問題覆蓋狀態，對未達標者於報告明確標註；成稿以 validate_citations
    驗證內文 [n]、移除越界編號；參考文獻由程式以真實標題生成。
  * 每個子問題前 / 各階段迴圈內即時檢查預算，達頂強制進入報告階段並標註不完整。
  * ResearchState 週期序列化（checkpoint）、asyncio.Semaphore 子問題層平行化。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import requests

from . import config
from . import prompts
from .agents import find_gaps, plan, read_source, run_searcher, verify
from .dedup import ContentDeduper
from .grounding import validate_citations
from .llm import complete_text
from .models import Budget, ResearchState

log = logging.getLogger("deep_research.orchestrator")


# ---------------------------------------------------------------------------
# 信心公式：Independence over quantity
# ---------------------------------------------------------------------------
def compute_confidence(independent_sources: int, avg_credibility: float, contradicted: bool) -> float:
    """獨立來源數為主因子；credibility 僅在 ≥2 獨立來源時作次要放大；矛盾折半。

    常數（0.4 上限 / 0.6:0.4 權重 / 飽和上限 3 / ×0.5）僅為起始值，應以人工標註校準。
    """
    if independent_sources < config.MIN_INDEPENDENT_SOURCES:
        # 單一來源（或 0）：信心上限壓在 0.4，杜絕「unverified 卻高信心」的矛盾。
        conf = min(0.4, 0.2 * max(independent_sources, 0))
    else:
        indep = min(independent_sources, config.CONFIDENCE_SATURATION) / config.CONFIDENCE_SATURATION
        conf = 0.6 * indep + 0.4 * indep * avg_credibility
    if contradicted:
        conf *= 0.5
    return round(min(conf, 1.0), 3)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(state: ResearchState, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
    log.info("已寫入 checkpoint：%s", path)


def load_checkpoint(path: str) -> ResearchState:
    with open(path, "r", encoding="utf-8") as f:
        state = ResearchState.from_dict(json.load(f))
    # 由既有來源文本重建去重器（deduper 不直接序列化）。
    state.deduper = ContentDeduper()
    for src in state.sources.values():
        state.deduper.is_duplicate(src.raw_text)
    log.info("已載入 checkpoint：%s（%d 來源）", path, len(state.sources))
    return state


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------
def _build_ref_map(state: ResearchState):
    """從實際被引用的來源，依數字排序 source id 產生引用編號。

    回傳 (source_id -> 編號, 編號 -> Source)。
    """
    cited_ids = set()
    for f in state.findings:
        cited_ids.update(f.supporting)
        cited_ids.update(f.conflicting)
    ordered = sorted(cited_ids)                      # 以數字排序，避免 s0,s1,s10,s2
    sid_to_num = {sid: i + 1 for i, sid in enumerate(ordered)}
    num_to_src = {i + 1: state.sources[sid] for i, sid in enumerate(ordered)}
    return sid_to_num, num_to_src


def synthesize(budget: Budget, state: ResearchState) -> str:
    sid_to_num, num_to_src = _build_ref_map(state)
    valid_numbers = set(num_to_src.keys())

    # 準備給 LLM 的發現清單：支持/反駁皆以引用編號呈現。
    lines = []
    for f in sorted(state.findings, key=lambda x: (x.subquestion_id, -x.confidence)):
        sup = " ".join(f"[{sid_to_num[s]}]" for s in f.supporting if s in sid_to_num)
        con = " ".join(f"[{sid_to_num[s]}]" for s in f.conflicting if s in sid_to_num)
        lines.append(
            f"- 陳述：{f.statement}\n  支持引用：{sup or '（無）'}"
            f"｜反駁引用：{con or '（無）'}｜信心：{f.confidence:.2f}"
        )
    findings_block = "\n".join(lines) if lines else "（無已驗證發現）"

    coverage_lines = [
        f"- [{'已覆蓋' if s.covered else '覆蓋不完整'} conf={s.confidence:.2f}] {s.text}"
        for s in state.subquestions
    ]
    coverage_block = "\n".join(coverage_lines)
    allowed = ", ".join(f"[{n}]" for n in sorted(valid_numbers)) or "（無可用引用）"

    user = (
        f"研究問題：{state.question}\n\n允許引用編號：{allowed}\n\n"
        f"已驗證發現：\n{findings_block}\n\n子問題覆蓋狀態：\n{coverage_block}\n\n"
        + ("注意：本研究因預算或覆蓋不足而不完整，請在報告中據實反映。\n" if state.incomplete else "")
        + "請據此撰寫繁體中文研究報告。"
    )

    report = complete_text(budget, prompts.SYNTH_SYS, user, max_tokens=config.SYNTH_MAX_TOKENS)

    # 內文引用程式驗證：移除越界 [n]
    report, used, invalid = validate_citations(report, valid_numbers)
    if invalid:
        log.warning("移除了越界引用編號：%s", sorted(invalid))

    # 由程式生成參考文獻（真實標題 + URL）
    refs = ["\n## 參考文獻"]
    for num in sorted(num_to_src.keys()):
        src = num_to_src[num]
        refs.append(f"[{num}] {src.title} — {src.url}")
    if len(refs) == 1:
        refs.append("（無被引用之來源）")

    # 覆蓋度附錄
    appendix = ["\n## 覆蓋度附錄"]
    if state.incomplete:
        appendix.append("> 本研究因預算耗盡或覆蓋不足而**不完整**。")
    for s in state.subquestions:
        mark = "已覆蓋" if s.covered else "覆蓋不完整"
        appendix.append(f"- {mark}（信心 {s.confidence:.2f}）：{s.text}")

    return report.strip() + "\n" + "\n".join(refs) + "\n" + "\n".join(appendix) + "\n"


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    coverage: float
    covered: int
    total_subquestions: int
    avg_independent_sources: float
    billed_tokens: float
    iterations: int
    n_sources: int
    n_findings: int
    incomplete: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _compute_metrics(state: ResearchState, budget: Budget) -> Metrics:
    total = len(state.subquestions) or 1
    covered = sum(1 for s in state.subquestions if s.covered)
    indeps = []
    for f in state.findings:
        indeps.append(len({state.sources[s].registered_domain
                           for s in f.supporting if s in state.sources}))
    avg_indep = sum(indeps) / len(indeps) if indeps else 0.0
    return Metrics(
        coverage=round(covered / total, 3),
        covered=covered,
        total_subquestions=len(state.subquestions),
        avg_independent_sources=round(avg_indep, 3),
        billed_tokens=round(budget.spent, 1),
        iterations=state.iterations,
        n_sources=len(state.sources),
        n_findings=len(state.findings),
        incomplete=state.incomplete,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _process_subquestion(search, state, budget, subq, session, allowlist, lock) -> None:
    """單一子問題：searcher 蒐集 → 逐來源 read。claim 寫入 state（以鎖序列化）。"""
    source_ids, covered = run_searcher(
        search, state, budget, subq, session=session, allowlist=allowlist, lock=lock
    )
    claims = []
    for sid in source_ids:
        if budget.exhausted:
            break
        claims.extend(read_source(budget, state, subq, state.sources[sid]))
    with lock:
        state.claims.extend(claims)
    log.info("子問題 #%d：收錄 %d 來源、%d claim（達獨立來源門檻=%s）",
             subq.id, len(source_ids), len(claims), covered)


def deep_research(
    question: str,
    *,
    search,
    budget: Optional[Budget] = None,
    state: Optional[ResearchState] = None,
    allowlist=None,
    checkpoint_path: Optional[str] = None,
    parallel: bool = False,
    concurrency: int = 4,
    max_subquestions: int = 8,
) -> dict:
    """對研究問題做多來源、多步驟研究並產出帶引用的報告。

    LLM / Embedding 呼叫走共用 provider 抽象層（part2/common）：規劃/閱讀/驗證/彙整用
    complete、搜尋 agent 用 run_tool_loop、搜尋結果用 Mistral embeddings 輕量排序。
    回傳 {'report', 'metrics', 'state'}。
    """
    budget = budget or Budget(total=config.DEFAULT_TOKEN_BUDGET)
    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT})

    if state is None:
        state = ResearchState(question=question)
        state.deduper = ContentDeduper()
        for text in plan(budget, question, max_subquestions=max_subquestions):
            state.add_subquestion(text)
    elif state.deduper is None:
        state.deduper = ContentDeduper()

    if not state.subquestions:
        log.warning("規劃未產生任何子問題，改用原問題作為單一子問題。")
        state.add_subquestion(question)

    try:
        for it in range(config.MAX_ITERATIONS):
            state.iterations = it + 1
            pending = [s for s in state.subquestions if not s.covered]
            if not pending:
                log.info("所有子問題已覆蓋，提前收斂。")
                break
            if budget.exhausted:
                state.incomplete = True
                log.warning("預算耗盡，強制進入報告階段並標註不完整。")
                break

            if parallel:
                _run_pending_parallel(search=search, state=state, budget=budget,
                                      pending=pending, session=session, allowlist=allowlist,
                                      concurrency=concurrency)
            else:
                lock = threading.Lock()
                for subq in pending:
                    if budget.exhausted:
                        state.incomplete = True
                        break
                    _process_subquestion(search, state, budget, subq, session, allowlist, lock)

            # 驗證：對每個子問題以「歷史+本輪全部 claim」重新分群
            for subq in state.subquestions:
                if budget.exhausted:
                    state.incomplete = True
                    break
                verify(budget, state, subq, compute_confidence)

            if checkpoint_path:
                save_checkpoint(state, checkpoint_path)

            # 覆蓋率達標即停；否則做缺口分析追加子問題
            covered = sum(1 for s in state.subquestions if s.covered)
            if state.subquestions and covered == len(state.subquestions):
                break
            if it < config.MAX_ITERATIONS - 1 and not budget.exhausted:
                for gap in find_gaps(budget, state):
                    state.add_subquestion(gap)

        if any(not s.covered for s in state.subquestions):
            state.incomplete = state.incomplete or budget.exhausted

        report = synthesize(budget, state)
    finally:
        session.close()

    metrics = _compute_metrics(state, budget)
    log.info("研究完成｜指標：%s", json.dumps(metrics.to_dict(), ensure_ascii=False))
    if checkpoint_path:
        save_checkpoint(state, checkpoint_path)

    return {"report": report, "metrics": metrics.to_dict(), "state": state}


# ---------------------------------------------------------------------------
# 平行化（asyncio.Semaphore 限流）
# ---------------------------------------------------------------------------
def _run_pending_parallel(*, search, state, budget, pending, session, allowlist, concurrency):
    """子問題層平行 + semaphore 限流；共享 ResearchState 以鎖序列化寫入。"""
    lock = threading.Lock()

    async def _driver():
        sem = asyncio.Semaphore(concurrency)

        async def handle(subq):
            async with sem:
                await asyncio.to_thread(
                    _process_subquestion, search, state, budget, subq, session, allowlist, lock,
                )

        await asyncio.gather(*(handle(sq) for sq in pending))

    asyncio.run(_driver())
