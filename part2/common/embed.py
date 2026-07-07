"""
Provider 無關的 Embeddings 介面（OpenAI 相容）。

預設用 Mistral 的 `mistral-embed`（1024 維、多語、含繁中）。Groq 不提供 embeddings，
故 RAG 類任務（文件問答、文件比較）的向量化改走此層的 embeddings 供應商。

環境變數：
  EMBED_PROVIDER (mistral | nvidia | together | openai)  預設 mistral
  EMBED_MODEL    覆寫預設嵌入模型
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

from openai import OpenAI

# 供應商 → (base_url, 金鑰環境變數, 預設嵌入模型)
EMBED_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "mistral":  ("https://api.mistral.ai/v1",           "MISTRAL_API_KEY",  "mistral-embed"),
    "nvidia":   ("https://integrate.api.nvidia.com/v1", "NVIDIA_NIM_API_KEY", "nvidia/nv-embedqa-e5-v5"),
    "together": ("https://api.together.xyz/v1",         "TOGETHER_API_KEY", "BAAI/bge-base-en-v1.5"),
    "openai":   ("https://api.openai.com/v1",           "OPENAI_API_KEY",   "text-embedding-3-small"),
}

_DEFAULT_PROVIDER = "mistral"
_MAX_BATCH = 32  # 單次送入的文字數上限，避免超過供應商限制

_client_cache: dict[str, tuple[OpenAI, str]] = {}


def _get_embed_client() -> tuple[OpenAI, str, str]:
    provider = os.getenv("EMBED_PROVIDER", _DEFAULT_PROVIDER).lower()
    if provider not in EMBED_PROVIDERS:
        raise ValueError(f"未知的 EMBED_PROVIDER={provider}，可選：{list(EMBED_PROVIDERS)}")
    base_url, key_env, default_model = EMBED_PROVIDERS[provider]
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(f"缺少 {key_env}（EMBED_PROVIDER={provider} 需要它）。")
    model = os.getenv("EMBED_MODEL", default_model)
    if provider not in _client_cache:
        _client_cache[provider] = (OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=3), model)
    client, _ = _client_cache[provider]
    return client, model, provider


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文字轉成向量。自動分批並含退避重試。回傳與輸入等長的向量列表。"""
    if not texts:
        return []
    client, model, provider = _get_embed_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), _MAX_BATCH):
        batch = texts[i:i + _MAX_BATCH]
        kwargs: dict[str, Any] = {"model": model, "input": batch}
        # NVIDIA 的非對稱模型需要 input_type
        if provider == "nvidia":
            kwargs["extra_body"] = {"input_type": "passage"}
        delay = 1.0
        for attempt in range(4):
            try:
                resp = client.embeddings.create(**kwargs)
                out.extend([d.embedding for d in resp.data])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                if not any(t in str(e).lower() for t in ("429", "rate", "timeout", "500", "502", "503")):
                    raise
                time.sleep(delay)
                delay *= 2
    return out


def embed_one(text: str) -> list[float]:
    """單一文字轉向量。"""
    return embed_texts([text])[0]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """兩向量的 cosine 相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
