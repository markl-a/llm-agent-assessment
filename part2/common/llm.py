"""
Provider 無關的 LLM 介面（OpenAI 相容）。

支援多家 OpenAI 相容供應商，預設 Groq（免費、快速、支援 tool calling）。
提供三種常用形態：
  - complete(system, user)              一次性文字回覆
  - stream_complete(system, user)       串流逐字回覆（generator）
  - run_tool_loop(system, msgs, tools)  tool-calling agent 迴圈（自動執行工具並回饋）

設計目標：讓四個 Part 2 專案共用同一套 LLM 呼叫路徑，只靠環境變數切換供應商 / 模型，
不必改動各專案的業務邏輯（RAG、對齊、去重、問卷…）。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Iterable

from openai import OpenAI

# 供應商 → (base_url, 金鑰環境變數, 預設模型)。皆為 OpenAI 相容端點。
LLM_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "groq":       ("https://api.groq.com/openai/v1",            "GROQ_API_KEY",       "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1",              "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct"),
    "cerebras":   ("https://api.cerebras.ai/v1",                "CEREBRAS_API_KEY",   "llama-3.3-70b"),
    "together":   ("https://api.together.xyz/v1",               "TOGETHER_API_KEY",   "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "nvidia":     ("https://integrate.api.nvidia.com/v1",       "NVIDIA_NIM_API_KEY", "meta/llama-3.3-70b-instruct"),
    "mistral":    ("https://api.mistral.ai/v1",                 "MISTRAL_API_KEY",    "mistral-large-latest"),
}

_DEFAULT_PROVIDER = "groq"

# 每個 (provider) 快取一個 client，避免重複建立
_clients: dict[str, OpenAI] = {}


def get_llm(provider: str | None = None) -> tuple[OpenAI, str]:
    """依環境變數建立（或取用快取的）OpenAI 相容 client，回傳 (client, model)。"""
    provider = (provider or os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER)).lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"未知的 LLM_PROVIDER={provider}，可選：{list(LLM_PROVIDERS)}")
    base_url, key_env, default_model = LLM_PROVIDERS[provider]
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(f"缺少 {key_env}（LLM_PROVIDER={provider} 需要它）。請設定環境變數或 .env。")
    model = os.getenv("LLM_MODEL", default_model)
    if provider not in _clients:
        _clients[provider] = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=3)
    return _clients[provider], model


def _create_with_retry(client: OpenAI, **kwargs) -> Any:
    """對暫時性錯誤（429 / 5xx）做指數退避重試；SDK 本身也有 max_retries 作為第二層保險。"""
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — 統一退避後重拋
            last_exc = e
            msg = str(e).lower()
            transient = any(t in msg for t in ("429", "rate", "timeout", "500", "502", "503", "overload"))
            if not transient or attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    raise last_exc  # pragma: no cover


def complete(system: str | None, user: str, *, temperature: float = 0.0,
             max_tokens: int = 1024, model: str | None = None,
             provider: str | None = None) -> str:
    """一次性文字回覆。"""
    client, m = get_llm(provider)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = _create_with_retry(
        client, model=model or m, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def chat(messages: list[dict], *, tools: list[dict] | None = None,
         tool_choice: str | None = None, temperature: float = 0.0,
         max_tokens: int = 1024, model: str | None = None,
         provider: str | None = None) -> Any:
    """低階：直接送一組 messages（可帶 tools），回傳原始 choices[0].message。"""
    client, m = get_llm(provider)
    kwargs: dict[str, Any] = dict(
        model=model or m, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    resp = _create_with_retry(client, **kwargs)
    return resp.choices[0].message


def stream_complete(system: str | None, user: str, *, temperature: float = 0.0,
                    max_tokens: int = 1024, model: str | None = None,
                    provider: str | None = None) -> Iterable[str]:
    """串流逐字回覆，yield 文字片段。"""
    client, m = get_llm(provider)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    stream = client.chat.completions.create(
        model=model or m, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True,
    )
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def run_tool_loop(
    system: str,
    messages: list[dict],
    tools: list[dict],
    tool_impls: dict[str, Callable[[dict], Any]],
    *,
    max_rounds: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model: str | None = None,
    provider: str | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> dict:
    """
    OpenAI 相容的 tool-calling agent 迴圈。

    - messages：初始對話（不含 system；system 另傳）。本函式會「就地」append 後續回合，
      因此呼叫端可用同一個 list 保存對話記憶。
    - tools：OpenAI function-calling 格式的工具規格。
    - tool_impls：name -> callable(arguments: dict) -> 任意可序列化結果。
    - on_event：可選 callback，接收 (event_type, payload)，event_type ∈ {"tool_call","tool_result","final"}。

    回傳 {"text": 最終回覆, "rounds": 使用回合數, "messages": 完整對話}。
    """
    client, m = get_llm(provider)
    full: list[dict] = [{"role": "system", "content": system}] + messages

    for round_i in range(max_rounds):
        resp = _create_with_retry(
            client, model=model or m, messages=full,
            tools=tools, tool_choice="auto",
            temperature=temperature, max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        # 把 assistant 這一輪原樣加回（含 tool_calls），維持對話一致性
        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        full.append(assistant_entry)

        if not msg.tool_calls:
            if on_event:
                on_event("final", {"text": msg.content or ""})
            messages[:] = full[1:]  # 回寫記憶（去掉 system）
            return {"text": (msg.content or "").strip(), "rounds": round_i + 1, "messages": full}

        # 依序執行每個 tool call，並把結果以 role=tool 回饋
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if on_event:
                on_event("tool_call", {"name": name, "arguments": args})
            impl = tool_impls.get(name)
            if impl is None:
                result: Any = {"error": f"未知工具 {name}"}
            else:
                try:
                    result = impl(args)
                except Exception as e:  # noqa: BLE001 — 工具錯誤回饋給模型而非中斷
                    result = {"error": f"{type(e).__name__}: {e}"}
            if on_event:
                on_event("tool_result", {"name": name, "result": result})
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            full.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    # 用盡回合仍未收斂：做一次無工具的收尾
    resp = _create_with_retry(
        client, model=model or m,
        messages=full + [{"role": "user", "content": "請根據以上資訊直接給出最終答覆，不要再呼叫工具。"}],
        temperature=temperature, max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    if on_event:
        on_event("final", {"text": text})
    messages[:] = full[1:]
    return {"text": text, "rounds": max_rounds, "messages": full}
