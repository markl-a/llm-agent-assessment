# -*- coding: utf-8 -*-
"""網址正規化、網域可信度評分與 SSRF 防護。

修正對照：
  * 2.4-6  canonical_url 保留 query，僅移除已知追蹤參數後排序重組，不整段清空。
  * 2.4-7  SSRF：僅 http/https、拒絕私網/迴環/link-local/保留位址、逐跳驗證 redirect。
  * 2.4-8  score_domain 以精確或帶點界線比對，避免 notarxiv.org / xxxbbc.com 命中。
  * 2.4-16 計算獨立性前正規化網域（去 www./m./amp.、去 port、小寫、取 eTLD+1）。
  * 2.4-41 nih.gov 由 .gov→0.9 涵蓋，tier2 不重複列，消除死規則。
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit, urljoin, parse_qsl, urlencode


class UnsafeURLError(Exception):
    pass


# ---------------------------------------------------------------------------
# 網域正規化
# ---------------------------------------------------------------------------
_SUBDOMAIN_PREFIXES = ("www.", "m.", "amp.")

# 常見兩段式公共後綴（近似 eTLD；非完整 PSL，但足以避免明顯的子網域灌水）。
_TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk",
    "com.tw", "org.tw", "gov.tw", "edu.tw", "net.tw", "idv.tw",
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "com.hk", "org.hk", "com.au", "net.au", "org.au", "gov.au", "edu.au",
    "co.jp", "or.jp", "ne.jp", "go.jp", "ac.jp",
    "co.kr", "or.kr", "go.kr",
    "com.br", "com.sg", "com.my",
}


def normalize_domain(host: str) -> str:
    """小寫、去尾點、去 port、去常見子網域前綴。"""
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return ""
    if ":" in host:
        host = host.split(":", 1)[0]
    for p in _SUBDOMAIN_PREFIXES:
        if host.startswith(p):
            host = host[len(p):]
            break
    return host


def registered_domain(host: str) -> str:
    """取 eTLD+1（近似）。用於計算「獨立來源網域數」，避免子網域膨脹（2.4-16）。"""
    host = normalize_domain(host)
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    last2 = ".".join(labels[-2:])
    if last2 in _TWO_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])
    return last2


# ---------------------------------------------------------------------------
# canonical_url：保留 query，只移除追蹤參數（2.4-6）
# ---------------------------------------------------------------------------
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_EXACT = {
    "gclid", "fbclid", "mc_eid", "mc_cid", "_hsenc", "_hsmi", "igshid",
    "yclid", "dclid", "msclkid", "ref", "ref_src", "spm", "scm",
}


def _keep_param(key: str) -> bool:
    k = key.lower()
    if k in _TRACKING_EXACT:
        return False
    return not any(k.startswith(p) for p in _TRACKING_PREFIXES)


def canonical_url(url: str) -> str:
    """正規化網址作為去重鍵：小寫 scheme/host、去預設 port、去片段、
    保留查詢參數但剝除追蹤參數並排序（?id=1 與 ?id=2 不會被誤判為同一頁）。"""
    parts = urlsplit((url or "").strip())
    scheme = (parts.scheme or "https").lower()
    host = normalize_domain(parts.hostname or "")
    netloc = host
    if parts.port and parts.port not in (80, 443):
        netloc = f"{host}:{parts.port}"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if _keep_param(k)]
    pairs.sort()
    query = urlencode(pairs)
    return urlunsplit((scheme, netloc, path, query, ""))


# ---------------------------------------------------------------------------
# 網域可信度評分（2.4-8 / 2.4-41）
# ---------------------------------------------------------------------------
DOMAIN_TIERS = {
    "blocklist": {"contentfarm.example"},                    # 0.1
    "tier1_tld": {".gov", ".edu"},                           # 0.9（依最後一段比對）
    "tier1_domain": {"who.int", "nature.com", "arxiv.org"},  # 0.9
    "tier2_domain": {"reuters.com", "bbc.com"},              # 0.75（nih.gov 已由 .gov→0.9 涵蓋）
}


def _domain_matches(domain: str, target: str) -> bool:
    domain = normalize_domain(domain)
    if target.startswith("."):                               # TLD 類：取最後一段比對（.gov → gov）
        return domain.rsplit(".", 1)[-1] == target[1:]
    return domain == target or domain.endswith("." + target)  # 具體網域：精確或帶點界線


def score_domain(domain: str) -> float:
    domain = normalize_domain(domain)
    if any(_domain_matches(domain, t) for t in DOMAIN_TIERS["blocklist"]):
        return 0.1
    if any(_domain_matches(domain, t) for t in DOMAIN_TIERS["tier1_domain"]):
        return 0.9
    if any(_domain_matches(domain, t) for t in DOMAIN_TIERS["tier1_tld"]):
        return 0.9
    if any(_domain_matches(domain, t) for t in DOMAIN_TIERS["tier2_domain"]):
        return 0.75
    return 0.5


# ---------------------------------------------------------------------------
# SSRF 防護（2.4-7）
# ---------------------------------------------------------------------------
def _ip_is_blocked(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def assert_safe_url(url: str, allowlist=None) -> None:
    """驗證 URL 可安全抓取，否則丟出 UnsafeURLError。

    allowlist 為 None 表示不啟用網域允許清單；否則需 registered_domain 命中。
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError(f"不支援的 scheme：{parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("缺少主機名稱")
    if allowlist is not None and registered_domain(host) not in allowlist:
        raise UnsafeURLError(f"網域不在允許清單：{host}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeURLError(f"DNS 解析失敗：{host} ({e})")
    if not infos:
        raise UnsafeURLError(f"無法解析主機：{host}")
    for info in infos:
        ip = info[4][0]
        if _ip_is_blocked(ip):
            raise UnsafeURLError(f"拒絕私網/保留位址：{host} -> {ip}")


def resolve_redirect(base_url: str, location: str) -> str:
    """把相對 Location 併回絕對 URL。"""
    return urljoin(base_url, location)
