"""Thin REST client for a self-hosted Letta server (tested against letta==0.16.8).

The grid worker runs in the experiment environment (anaconda + PYTHONNOUSERSITE=1),
which cannot import the ``letta`` / ``letta_client`` packages (numpy 2.x and
pydantic pins conflict with the rest of the stack), so the backend talks to the
server over HTTP with ``requests``. Only the handful of endpoints the
``LettaBackend`` needs are wrapped:

- ``POST /v1/agents/``                              create an agent with memory blocks
- ``POST /v1/agents/{id}/messages``                 send one user message (agent runs its loop)
- ``GET  /v1/agents/{id}/core-memory/blocks``       read the self-edited core memory
- ``GET  /v1/agents/{id}/archival-memory/search``   semantic search over archival memory
- ``GET  /v1/agents/{id}/archival-memory``          list passages (for counts)
- ``DELETE /v1/agents/{id}``

Server side (see README.md): a separate virtual environment holds letta 0.16.8 pinned
to its release date with ``uv --exclude-newer``; it needs PostgreSQL + pgvector
(``LETTA_PG_URI``) and an OpenAI-compatible embedding endpoint
(``scripts/local_embedding_server.py``) because vLLM only serves chat.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

DEFAULT_BASE_URL = os.environ.get("LETTA_BASE_URL", "http://127.0.0.1:8283")
DEFAULT_EMBED_ENDPOINT = os.environ.get("LETTA_EMBED_ENDPOINT", "http://127.0.0.1:8285/v1")
DEFAULT_EMBED_MODEL = os.environ.get("LETTA_EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_EMBED_DIM = int(os.environ.get("LETTA_EMBED_DIM", "384"))
# 16k: Letta summarises older in-context messages once the window fills, which
# keeps each agent step's prompt (persona + human blocks + recent emails) bounded.
DEFAULT_CONTEXT_WINDOW = int(os.environ.get("LETTA_CONTEXT_WINDOW", "16000"))
DEFAULT_MAX_STEPS = int(os.environ.get("LETTA_MAX_STEPS", "3"))
DEFAULT_TIMEOUT = float(os.environ.get("LETTA_HTTP_TIMEOUT", "1800"))

# Tool names whose invocation means the agent wrote to memory.
MEMORY_WRITE_TOOLS = {
    "core_memory_append",
    "core_memory_replace",
    "memory",
    "memory_apply_patch",
    "memory_insert",
    "memory_replace",
    "memory_rethink",
    "archival_memory_insert",
}


class LettaServerError(RuntimeError):
    pass


class LettaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._s = requests.Session()

    # ---- low level -------------------------------------------------------
    def _req(self, method: str, path: str, *, retries: int = 3, **kw) -> Any:
        url = f"{self.base_url}{path}"
        kw.setdefault("timeout", self.timeout)
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = self._s.request(method, url, **kw)
            except requests.RequestException as exc:  # connection-level
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code >= 500 and attempt < retries - 1:
                last = LettaServerError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise LettaServerError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
            if r.status_code == 204 or not r.content:
                return None
            return r.json()
        raise LettaServerError(f"{method} {path} failed after {retries} attempts: {last}")

    def health(self) -> dict:
        return self._req("GET", "/v1/health/")

    # ---- agents ----------------------------------------------------------
    def create_agent(
        self,
        *,
        name: str,
        persona_text: str,
        human_text: str,
        model: str,
        model_endpoint: str,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        embed_endpoint: str = DEFAULT_EMBED_ENDPOINT,
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed_dim: int = DEFAULT_EMBED_DIM,
        system: str | None = None,
        tags: list[str] | None = None,
        agent_type: str | None = None,
        tool_rules: list[dict] | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "name": name,
            "memory_blocks": [
                {"label": "persona", "value": persona_text},
                {"label": "human", "value": human_text},
            ],
            "llm_config": {
                "model": model,
                "model_endpoint_type": "openai",
                "model_endpoint": model_endpoint,
                "context_window": context_window,
                "put_inner_thoughts_in_kwargs": True,
                "temperature": 0.0,
                "enable_reasoner": False,
                "handle": f"openai/{model}",
            },
            "embedding_config": {
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": embed_endpoint,
                "embedding_model": embed_model,
                "embedding_dim": embed_dim,
                "embedding_chunk_size": 300,
                "handle": f"openai/{embed_model}",
            },
            "include_base_tools": True,
            "enable_reasoner": False,
            "reasoning": False,
            "tags": tags or [],
        }
        if system:
            body["system"] = system
        if agent_type:
            body["agent_type"] = agent_type
        if tool_rules:
            body["tool_rules"] = tool_rules
        return self._req("POST", "/v1/agents/", json=body)

    def delete_agent(self, agent_id: str) -> None:
        try:
            self._req("DELETE", f"/v1/agents/{agent_id}", retries=1)
        except LettaServerError:
            pass

    def list_tools(self, agent_id: str) -> list[str]:
        tools = self._req("GET", f"/v1/agents/{agent_id}/tools") or []
        return [t.get("name", "") for t in tools]

    # ---- messages --------------------------------------------------------
    def send_message(self, agent_id: str, text: str, *, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
        body = {
            "messages": [{"role": "user", "content": text}],
            "max_steps": max_steps,
            "streaming": False,
        }
        return self._req("POST", f"/v1/agents/{agent_id}/messages", json=body, retries=2)

    @staticmethod
    def memory_tool_calls(response: dict) -> list[str]:
        """Names of memory-writing tools invoked in a send_message response."""
        names: list[str] = []
        for m in (response or {}).get("messages", []) or []:
            mtype = m.get("message_type") or m.get("type") or ""
            if "tool_call" not in mtype:
                continue
            calls = m.get("tool_calls") or ([m["tool_call"]] if m.get("tool_call") else [])
            for c in calls:
                n = c.get("name") or (c.get("function") or {}).get("name") or ""
                if n in MEMORY_WRITE_TOOLS:
                    names.append(n)
        return names

    # ---- memory reads ----------------------------------------------------
    def core_blocks(self, agent_id: str) -> list[dict]:
        return self._req("GET", f"/v1/agents/{agent_id}/core-memory/blocks") or []

    def archival_search(self, agent_id: str, query: str, *, top_k: int = 10) -> list[dict]:
        res = self._req(
            "GET",
            f"/v1/agents/{agent_id}/archival-memory/search",
            params={"query": query, "top_k": top_k},
        ) or {}
        return res.get("results", []) if isinstance(res, dict) else list(res)

    def archival_count(self, agent_id: str) -> int:
        rows = self._req("GET", f"/v1/agents/{agent_id}/archival-memory", params={"limit": 1000}) or []
        return len(rows)


def render_core_memory(blocks: list[dict]) -> str:
    """Human-readable dump of the agent's core memory blocks (label: value)."""
    parts = []
    for b in blocks:
        label = b.get("label") or "block"
        value = (b.get("value") or "").strip()
        if value:
            parts.append(f"[{label}]\n{value}")
    return "\n\n".join(parts)
