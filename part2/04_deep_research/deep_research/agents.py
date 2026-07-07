# -*- coding: utf-8 -*-
"""各研究階段：plan / run_searcher / read_source / verify / find_gaps。

LLM / Embedding 走共用 provider 抽象層（part2/common）：
  * plan / read_source / verify / find_gaps → llm.call_json（走 common.complete + 穩健 JSON 解析）
  * run_searcher（web_search / fetch_page / done）→ llm.tool_loop（走 common.run_tool_loop）
  * 搜尋結果以 llm.embed_rerank（Mistral embeddings）做輕量向量排序，缺金鑰時自動降級。

核心業務邏輯與價值：
  * 獨立來源以正規化網域（eTLD+1）計數；run_searcher 收尾強制檢查獨立網域數，不足則由
    done 工具回覆提示續搜（在工具迴圈內落地，而非只在 prompt 提示）。
  * Reader 收到的正文以 <external_content> 包裹（抗間接注入）；抽取的 quote 需通過
    quote_supported 才保留（抗幻覺）。
  * 抓取做 SSRF 防護、canonical_url 去重、MinHash 內容去重。
  * searcher/reader/verify/gap 迴圈內即時檢查預算，達頂即停。

訊息一致性（tool_call / tool_result 的對應）由 common.run_tool_loop 內部維護。
"""
from __future__ import annotations

import contextlib
import json
import logging
from typing import List, Optional, Tuple

import requests

from . import config
from . import prompts
from .dedup import cluster_texts
from .grounding import quote_supported, wrap_untrusted
from .llm import call_json, embed_rerank, tool_loop
from .models import Budget, Claim, ResearchState, Source, SubQuestion, VerifiedFinding
from .net import canonical_url, normalize_domain, registered_domain, score_domain, UnsafeURLError
from .search import FetchError, SearchProvider, fetch_and_extract

log = logging.getLogger("deep_research.agents")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
def plan(budget: Budget, question: str, *, max_subquestions: int = 8) -> List[str]:
    """把研究問題拆成子問題（回傳文字列表）。"""
    data = call_json(
        budget,
        prompts.PLANNER_SYS,
        f"研究問題：{question}",
        max_tokens=config.PLAN_MAX_TOKENS,
    )
    subs = [s.get("text", "").strip() for s in data.get("subquestions", []) if isinstance(s, dict)]
    subs = [s for s in subs if s]
    return subs[:max_subquestions]


# ---------------------------------------------------------------------------
# Searcher（tool-use 迴圈，走 common.run_tool_loop）
# ---------------------------------------------------------------------------
def _independent_domains(state: ResearchState, source_ids: List[int]) -> int:
    return len({state.sources[sid].registered_domain for sid in source_ids})


def _make_source(state: ResearchState, url: str, text: str, title: str) -> Source:
    domain = normalize_domain((url.split("//", 1)[-1].split("/", 1)[0]))
    src = Source(
        id=state.new_source_id(),
        url=url,
        canonical_url=canonical_url(url),
        domain=domain,
        registered_domain=registered_domain(domain),
        title=(title or domain or url)[:300],           # 真實標題，退回 domain
        credibility=score_domain(domain),
        raw_text=(text or "")[: config.MAX_SOURCE_CHARS],  # 可驗證範圍
    )
    return src


def _fetch_into_state(
    url: str,
    state: ResearchState,
    collected: List[int],
    session: requests.Session,
    allowlist,
    lock,
) -> str:
    """抓取 URL、去重後收錄為 Source，回傳給模型的工具結果文字。"""
    canon = canonical_url(url)
    with lock:
        if canon in state.seen_canonical:
            return f"已收錄過相同網址（{canon}），略過。"
    text, title = fetch_and_extract(url, session=session, allowlist=allowlist)
    if not text or len(text.strip()) < 80:
        return "抓取成功但正文過短或為空，未收錄。"
    with lock:
        if canon in state.seen_canonical:               # 二次檢查避免競態
            return "已被其他流程收錄，略過。"
        if state.deduper is not None and state.deduper.is_duplicate(text):
            state.seen_canonical.add(canon)
            return "內容與既有來源近似重複，未收錄。"
        src = _make_source(state, url, text, title)
        state.sources[src.id] = src
        state.seen_canonical.add(canon)
        collected.append(src.id)
    return (
        f"已收錄來源 #{src.id}｜標題：{src.title}｜網域：{src.domain}"
        f"（可信度 {src.credibility}）｜正文預覽：{text[:300]}"
    )


def run_searcher(
    search: SearchProvider,
    state: ResearchState,
    budget: Budget,
    subq: SubQuestion,
    session: Optional[requests.Session] = None,
    allowlist=None,
    lock=None,
) -> Tuple[List[int], bool]:
    """對單一子問題執行 tool-use 迴圈，回傳 (收錄的 source ids, 是否達到獨立來源門檻)。

    工具實作以閉包持有 state/collected/session/lock；done 工具在迴圈內實際檢查獨立
    網域數，不足則回覆提示要求續搜（把「強制檢查」落地在工具回饋，而非只靠 prompt）。
    """
    lock = lock or contextlib.nullcontext()
    session = session or requests.Session()
    collected: List[int] = []

    def _web_search(args: dict):
        query = (args.get("query") or "").strip()
        if not query:
            return {"error": "缺少參數 'query'"}
        hits = search.search(query, config.MAX_RESULTS_PER_SEARCH)
        if not hits:
            return {"results": [], "note": "查無結果或搜尋暫時失敗，請換關鍵字或（來源已足時）呼叫 done。"}
        if config.USE_EMBED_RERANK:
            hits = embed_rerank(subq.text, hits)         # Mistral 向量排序（可降級）
        return {"results": hits}

    def _fetch_page(args: dict):
        url = (args.get("url") or "").strip()
        if not url:
            return {"error": "缺少參數 'url'"}
        if len(collected) >= config.MAX_SOURCES_PER_SUBQ:
            return {"note": "本子問題來源已足夠，請呼叫 done。"}
        try:
            return {"result": _fetch_into_state(url, state, collected, session, allowlist, lock)}
        except UnsafeURLError as e:
            return {"error": f"拒絕不安全的網址（{e}）"}
        except FetchError as e:
            return {"error": f"抓取失敗（{e}）"}
        except Exception as e:  # noqa: BLE001 — 工具例外回饋給模型而非中斷
            return {"error": f"{type(e).__name__}: {e}"}

    def _done(args: dict):
        indep = _independent_domains(state, collected)
        if indep < config.MIN_INDEPENDENT_SOURCES and not budget.exhausted:
            return {
                "status": "insufficient",
                "note": (f"目前僅 {indep} 個獨立網域（需 ≥{config.MIN_INDEPENDENT_SOURCES}），"
                         f"請再用 web_search / fetch_page 補充不同網域的來源後再呼叫 done。"),
            }
        return {"status": "ok", "reason": (args.get("reason") or "").strip() or "來源蒐集完成。"}

    tool_impls = {"web_search": _web_search, "fetch_page": _fetch_page, "done": _done}

    messages = [{
        "role": "user",
        "content": (
            f"總研究問題：{state.question}\n本次子問題：{subq.text}\n"
            f"請蒐集至少 {config.MIN_INDEPENDENT_SOURCES} 個不同網域的可靠來源，完成後呼叫 done。"
        ),
    }]

    def _on_event(ev_type: str, payload: dict) -> None:
        if ev_type == "tool_call":
            args_preview = json.dumps(payload.get("arguments", {}), ensure_ascii=False)[:140]
            log.info("searcher 子問題#%d → %s(%s)", subq.id, payload.get("name"), args_preview)

    if not budget.exhausted:
        try:
            tool_loop(
                budget,
                prompts.SEARCHER_SYS,
                messages,
                prompts.SEARCHER_TOOLS,
                tool_impls,
                max_rounds=config.MAX_SEARCHER_STEPS,
                max_tokens=config.STEP_MAX_TOKENS,
                on_event=_on_event,
            )
        except Exception as e:  # noqa: BLE001
            # 某些 OpenAI 相容供應商（如 Groq 的 llama）偶爾會產生格式不合的 tool_call，
            # 導致 400 tool_use_failed 並中斷迴圈。此處降級為「直接搜尋+抓取」備援，
            # 確保仍能蒐集到來源（common 層為共用/已驗證，不在此處修改其重試策略）。
            log.warning("搜尋 agent（tool-calling）中斷，改用直接搜尋+抓取備援：%s", e)

    # 備援 / 補強：若獨立網域仍不足，直接以子問題文字搜尋並抓取前幾筆補齊。
    if _independent_domains(state, collected) < config.MIN_INDEPENDENT_SOURCES and not budget.exhausted:
        _fallback_search_fetch(search, state, subq, collected, session, allowlist, lock)

    covered = _independent_domains(state, collected) >= config.MIN_INDEPENDENT_SOURCES
    if not covered:
        log.info("子問題 #%d 獨立來源不足（%d 個網域），標記覆蓋不足。",
                 subq.id, _independent_domains(state, collected))
    return collected, covered


def _fallback_search_fetch(search, state, subq, collected, session, allowlist, lock) -> None:
    """不依賴 LLM tool-calling 的備援：直接搜尋子問題並抓取前幾筆到 state。

    仍走既有的向量 rerank、SSRF 防護、canonical/內容去重邏輯，只是把「該搜什麼、
    該抓哪些」改為決定性策略，避免供應商的 tool-calling 不穩定造成整體無來源。
    """
    try:
        hits = search.search(subq.text, config.MAX_RESULTS_PER_SEARCH)
    except Exception as e:  # noqa: BLE001
        log.warning("備援搜尋失敗（子問題 #%d）：%s", subq.id, e)
        return
    if config.USE_EMBED_RERANK:
        hits = embed_rerank(subq.text, hits)
    for h in hits:
        if len(collected) >= config.MAX_SOURCES_PER_SUBQ:
            break
        if _independent_domains(state, collected) >= config.MIN_INDEPENDENT_SOURCES \
                and len(collected) >= config.MIN_INDEPENDENT_SOURCES:
            break
        url = h.get("url")
        if not url:
            continue
        try:
            msg = _fetch_into_state(url, state, collected, session, allowlist, lock)
            log.info("備援抓取 子問題#%d：%s → %s", subq.id, url[:80], msg[:60])
        except (UnsafeURLError, FetchError) as e:
            log.debug("備援抓取略過（%s）：%s", url[:80], e)
        except Exception as e:  # noqa: BLE001
            log.debug("備援抓取失敗（%s）：%s", url[:80], e)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def read_source(
    budget: Budget,
    state: ResearchState,
    subq: SubQuestion,
    source: Source,
) -> List[Claim]:
    """從單一來源抽取 claim；quote 需通過來源一致性校驗才保留（抗幻覺/抗注入）。"""
    if budget.exhausted:
        return []
    user = (
        f"子問題：{subq.text}\n以下為來源 #{source.id} 的正文：\n"
        + wrap_untrusted(source.raw_text)
    )
    try:
        data = call_json(budget, prompts.READER_SYS, user, max_tokens=config.READER_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001 — 解析失敗 → 跳過此來源，不崩潰
        log.warning("read_source 失敗（source #%d）：%s", source.id, e)
        return []

    claims: List[Claim] = []
    for item in data.get("claims", []):
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        quote = (item.get("quote") or "").strip()
        if not text or not quote:
            continue
        if not quote_supported(quote, source.raw_text):   # 未命中即丟棄
            log.debug("丟棄未通過 quote-grounding 的 claim（source #%d）", source.id)
            continue
        claims.append(Claim(
            id=state.new_claim_id(),
            subquestion_id=subq.id,
            source_id=source.id,
            text=text,
            quote=quote,
        ))
    return claims


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
def _cluster_key(claim_ids: List[int]) -> str:
    return "c:" + ",".join(str(i) for i in sorted(claim_ids))


def verify(
    budget: Budget,
    state: ResearchState,
    subq: SubQuestion,
    compute_confidence,
) -> None:
    """對某子問題的「歷史+本輪全部 claim」重新分群並交叉驗證。

    以每群成員為鍵做快取，重跑不重複燒 token；結果更新回 state.findings。
    """
    subq_claims = [c for c in state.claims if c.subquestion_id == subq.id]
    if not subq_claims:
        return

    clusters = cluster_texts([c.text for c in subq_claims], config.CLUSTER_JACCARD_THRESHOLD)

    # 先移除本子問題的舊 findings，稍後以最新分群結果重建（避免同事實重複 finding）。
    state.findings = [f for f in state.findings if f.subquestion_id != subq.id]
    best_conf = 0.0

    for idx_group in clusters:
        group_claims = [subq_claims[i] for i in idx_group]
        claim_ids = [c.id for c in group_claims]
        key = _cluster_key(claim_ids)

        cached = state.verify_cache.get(key)
        if cached is not None:
            finding = VerifiedFinding.from_dict(cached)
        else:
            finding = _verify_cluster(budget, state, subq, group_claims, key, compute_confidence)
            if finding is None:
                continue
            state.verify_cache[key] = finding.to_dict()

        state.findings.append(finding)
        best_conf = max(best_conf, finding.confidence)

    subq.confidence = best_conf
    subq.covered = best_conf >= config.HIGH_CONFIDENCE


def _verify_cluster(budget, state, subq, group_claims, key, compute_confidence):
    listing = "\n".join(
        f"[{i}] （來源 #{c.source_id}，網域 {state.sources[c.source_id].domain}）{c.text}"
        for i, c in enumerate(group_claims)
    )
    user = f"子問題：{subq.text}\n候選 claim：\n{wrap_untrusted(listing)}"
    try:
        data = call_json(budget, prompts.VERIFIER_SYS, user, max_tokens=config.VERIFY_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001
        log.warning("verify 分群失敗（%s）：%s", key, e)
        return None

    statement = (data.get("statement") or "").strip()
    if not statement:
        return None

    supporting_srcs, conflicting_srcs = set(), set()
    for lab in data.get("labels", []):
        if not isinstance(lab, dict):
            continue
        i = lab.get("index")
        stance = lab.get("stance")
        if not isinstance(i, int) or not (0 <= i < len(group_claims)):
            continue
        sid = group_claims[i].source_id
        if stance == "supports":
            supporting_srcs.add(sid)
        elif stance == "refutes":                     # conflicting 真正被填入
            conflicting_srcs.add(sid)

    if not supporting_srcs:                            # 無任何支持證據則不成立
        return None

    indep = len({state.sources[s].registered_domain for s in supporting_srcs})
    avg_cred = sum(state.sources[s].credibility for s in supporting_srcs) / len(supporting_srcs)
    contradicted = len(conflicting_srcs) > 0
    conf = compute_confidence(indep, avg_cred, contradicted)

    return VerifiedFinding(
        subquestion_id=subq.id,
        statement=statement,
        supporting=sorted(supporting_srcs),
        conflicting=sorted(conflicting_srcs),
        confidence=conf,
        contradicted=contradicted,
        cluster_key=key,
    )


# ---------------------------------------------------------------------------
# Gap 分析
# ---------------------------------------------------------------------------
def find_gaps(budget: Budget, state: ResearchState) -> List[str]:
    if budget.exhausted:
        return []
    status = "\n".join(
        f"- [{'已覆蓋' if s.covered else '未覆蓋'} conf={s.confidence:.2f}] {s.text}"
        for s in state.subquestions
    )
    findings = "\n".join(f"- {f.statement}" for f in state.findings[:20])
    user = f"研究問題：{state.question}\n子問題覆蓋狀態：\n{status}\n已確立發現：\n{findings}"
    try:
        data = call_json(budget, prompts.GAP_SYS, user, max_tokens=config.GAP_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001
        log.warning("find_gaps 失敗：%s", e)
        return []
    return [s.get("text", "").strip() for s in data.get("subquestions", [])
            if isinstance(s, dict) and s.get("text")][:3]
