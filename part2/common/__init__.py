"""
共用的 Provider 抽象層 —— 讓四個 Part 2 實作能在不同 LLM / Embedding 供應商間切換。

- llm.py    : OpenAI 相容的 Chat / Tool-calling / Streaming（預設 Groq）
- embed.py  : OpenAI 相容的 Embeddings（預設 Mistral）與向量工具

所有供應商皆走 OpenAI 相容 API，透過環境變數選擇：
  LLM_PROVIDER   (groq | openrouter | cerebras | together | nvidia | mistral)   預設 groq
  LLM_MODEL      覆寫預設模型
  EMBED_PROVIDER (mistral | nvidia | together | openai)                          預設 mistral
  EMBED_MODEL    覆寫預設嵌入模型
金鑰各自來自對應的 *_API_KEY 環境變數。
"""
from .llm import chat, complete, stream_complete, run_tool_loop, get_llm, LLM_PROVIDERS
from .embed import embed_texts, embed_one, cosine_sim, EMBED_PROVIDERS

__all__ = [
    "chat", "complete", "stream_complete", "run_tool_loop", "get_llm", "LLM_PROVIDERS",
    "embed_texts", "embed_one", "cosine_sim", "EMBED_PROVIDERS",
]
