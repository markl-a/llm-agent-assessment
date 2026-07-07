"""非同步 token-bucket 限流（TPM / RPM）。

僅設併發上限並不足夠：正式環境應依速率表節流，對 LLM 與 Embedding 兩端分別限流
（對應 review 2.2-21）。此處提供一個輕量、無外部相依的 async token-bucket，供 map /
reduce / embed 呼叫前 `await limiter.acquire(tokens)`。

實作為經典 token-bucket：以固定速率補充令牌，桶滿即封頂；請求需要的令牌不足時 sleep
到補足為止。RPM 與 TPM 各一個桶，任一不足都會阻塞。
"""

from __future__ import annotations

import asyncio
import time


class _Bucket:
    def __init__(self, rate_per_min: float, capacity: float | None = None) -> None:
        self.rate = max(rate_per_min, 1e-9) / 60.0  # 每秒補充量
        self.capacity = capacity if capacity is not None else max(rate_per_min, 1.0)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float) -> None:
        # 單次需求超過整桶容量時，封頂為容量，避免永遠等不到。
        amount = min(amount, self.capacity)
        while True:
            async with self._lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.updated) * self.rate
                )
                self.updated = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                deficit = amount - self.tokens
                wait = deficit / self.rate
            await asyncio.sleep(wait)


class RateLimiter:
    """同時受 RPM 與 TPM 限制的限流器。"""

    def __init__(self, rpm: int, tpm: int) -> None:
        self._req = _Bucket(rpm)
        self._tok = _Bucket(tpm)

    async def acquire(self, tokens: int) -> None:
        # 先請求數、再 token 數；兩者皆滿足才放行。
        await self._req.acquire(1)
        await self._tok.acquire(max(tokens, 1))
