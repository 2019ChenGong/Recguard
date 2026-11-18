"""
Unified async interface to an OpenAI-compatible LLM endpoint
(e.g. vLLM server).

Mirrors the style of the nanobot AgentInterface used in the MIA study.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx


class AgentInterface:
    """Thin async client for a remote OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "token-vllm",
        model: str = (
            "/bigtemp/trv3px/model_checkpoints/"
            "models--Qwen--Qwen2.5-72B-Instruct/"
            "snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
        ),
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = httpx.AsyncClient(timeout=timeout)
        self.query_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single-turn chat completion. Returns the assistant content string."""
        self.query_count += 1

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = await self.client.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AgentInterface":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
